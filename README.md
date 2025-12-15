# Caia Model Registry

Small FastAPI + SQLite model registry meant to stay hackable.

## What This Is

- A tiny registry for model artifacts + metadata (SQLite) with an audit trail (`model_events`)
- A place for eval runners to write suite-scoped metrics (`metrics.suites.<suite>`)
- A promotion workflow (`experimental` → `staging` → `production`) with optional gates

## What This Isn’t

- An artifact store (no uploads); store artifacts in S3/GCS/etc and reference them via `artifact_uri`
- A model runner/inference server (no checkpoint loading; no `torch.load()`)
- A full MLOps platform (intentionally minimal)

## 5-Minute Quickstart

Prereqs: Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env
# edit .env and set CAIA_REGISTRY_API_KEY
set -a; source .env; set +a

uvicorn server:app --host 127.0.0.1 --port 8001
```

In another terminal:

```bash
curl http://127.0.0.1:8001/health
curl -sS -H "X-API-Key: $CAIA_REGISTRY_API_KEY" "http://127.0.0.1:8001/models?limit=50&sort=updated_at&order=desc"
python smoke_test.py
```

## Deployment

This registry uses SQLite. For production, prefer a single-process server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1
```

Notes:
- WAL mode is enabled automatically, but SQLite still has a single-writer bottleneck.
- Avoid multiple workers for write-heavy workloads; if you need horizontal scaling, move to a client/server DB.

## Environment Variables

- `CAIA_REGISTRY_API_KEY` or `CAIA_REGISTRY_API_KEYS` (comma-separated): required for all endpoints except `/` and `/health` (server returns `503` if not configured).
- `CAIA_REGISTRY_TRUST_ACTOR_HEADER`: if `1`, trust `X-Actor` for audit logs; default `0` (otherwise actor is derived from the API key prefix).
- `CAIA_REGISTRY_CORS_ORIGINS` (comma-separated): allowed browser origins; if unset defaults to localhost-only; set to empty to allow none.
- `CAIA_REGISTRY_DB_PATH`: SQLite file path; default is `registry.db` next to `server.py`.
- `CAIA_REGISTRY_SQLITE_BUSY_TIMEOUT_MS`: SQLite busy timeout in ms (default `5000`).
- `CAIA_REGISTRY_MAX_HASH_BYTES`: max bytes the server will hash for local files (default `268435456`); larger files skip hashing but can still accept client-provided hash/size.
- `CAIA_REGISTRY_MAX_METRICS_BODY_BYTES`: max bytes accepted by `POST /models/{id}/metrics` (default `65536`).
- `CAIA_REGISTRY_MAX_EVENT_PAYLOAD_BYTES`: max bytes stored in `model_events.payload_json` (default `65536`).
- `CAIA_REGISTRY_MAX_WRITE_BODY_BYTES`: max bytes accepted by other write endpoints like `POST /models` and `PATCH /models/{id}` (default `262144`).
- `CAIA_REGISTRY_ALLOW_LOCAL_PATHS`: if `1`, allow request fields `checkpoint_path`/`config_path` as local filesystem paths (dev only; default `0`).
- `CAIA_REGISTRY_EXPOSE_LOCAL_PATHS`: if `1`, responses include `local_checkpoint_path`/`local_config_path` (default `0`).
- `CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE`: if set (e.g. `core-v1`), require suite exists in `metrics.suites[...]` to promote to production; blocks if candidate `score` is worse than current production when both numeric.

## API Examples (curl)

```bash
export CAIA_REGISTRY_API_KEY=dev

curl http://127.0.0.1:8001/health

curl -sS -H "X-API-Key: dev" "http://127.0.0.1:8001/models?limit=50&sort=updated_at&order=desc"
```

Register (preferred: client computes hashes/sizes; server never unpickles checkpoints):

```bash
curl -sS -H "X-API-Key: dev" -H "Content-Type: application/json" \
  -d '{
    "name": "demo-model",
    "version": "v1",
    "artifact_uri": "s3://bucket/path/to/checkpoint.pt",
    "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "checkpoint_size_bytes": 123,
    "config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "config_size_bytes": 456,
    "tags": ["demo", "experimental"]
  }' \
  http://127.0.0.1:8001/models
```

Write suite-scoped metrics (merges into `metrics.suites.<suite>`):

```bash
curl -sS -H "X-API-Key: dev" -H "Content-Type: application/json" \
  -d '{"suite":"smoke-v1","metrics":{"score":1.0,"config_parse_ok":true}}' \
  http://127.0.0.1:8001/models/1/metrics
```

Optional optimistic locking (reject with `409` if the model changed since you last read it):

```bash
UPDATED_AT=$(curl -sS -H "X-API-Key: dev" http://127.0.0.1:8001/models/1 | python -c 'import json,sys; print(json.load(sys.stdin)["updated_at"])')
curl -sS -H "X-API-Key: dev" -H "If-Updated-At: $UPDATED_AT" -H "Content-Type: application/json" \
  -d '{"suite":"smoke-v1","metrics":{"score":1.0}}' \
  http://127.0.0.1:8001/models/1/metrics
```

Read metrics:

```bash
curl -sS -H "X-API-Key: dev" "http://127.0.0.1:8001/models/1/metrics"
curl -sS -H "X-API-Key: dev" "http://127.0.0.1:8001/models/1/metrics?suite=smoke-v1"
```

Promote:

```bash
curl -sS -X POST -H "X-API-Key: dev" "http://127.0.0.1:8001/models/1/promote?to_status=production"
```

Events:

```bash
curl -sS -H "X-API-Key: dev" "http://127.0.0.1:8001/models/1/events?limit=50"
```

## CLIs

Register artifacts: `ingest.py`

```bash
python ingest.py \
  --registry-url http://127.0.0.1:8001 \
  --api-key dev \
  --name demo-model \
  --version v1 \
  --checkpoint-path ./checkpoints/checkpoint.pt \
  --config-path ./config.json
```

Run a lightweight benchmark and write metrics: `bench.py`

```bash
python bench.py \
  --registry-url http://127.0.0.1:8001 \
  --api-key dev \
  --model-id 1 \
  --suite smoke-v1 \
  --notes "local smoke"
```

DB maintenance: `maintenance.py`

```bash
python maintenance.py --db-path ./registry.db --backup ./registry.backup.db --vacuum
```

## License

MIT licensed. See `LICENSE`.

## OSS Release Checklist

- Run QA: `make qa` (expect exit code `0`) and review `qa_artifacts/qa_report.json`.
- Run “CI locally” equivalent: `make test` (or `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`).
- Secret/path scan: confirm no `.env`, `*.db`, `*.sqlite*`, `qa_artifacts/`, or local paths (e.g. `/Users/...`) are committed.
- Version + changelog: update `CHANGELOG.md` and `server.py` version together.
- Tag + push:
  - `git tag -a v0.2.0 -m "v0.2.0"`
  - `git push origin main --tags`
- Create GitHub release:
  - Draft a new release from tag `v0.2.0`
  - Paste highlights from `CHANGELOG.md`
