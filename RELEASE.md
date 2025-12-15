# Release Guide (v0.2.0)

This repo is intended to be published as `caiatech-model-registry` under MIT.

## Preflight

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
```

Run QA (should exit `0`):

```bash
make qa
```

Run the pre-release audit:

```bash
python scripts/pre_release_audit.py
```

## Tagging

Make sure versions match:
- `server.py` app version
- `CHANGELOG.md` top entry

Tag and push:

```bash
git tag -a v0.2.0 -m "v0.2.0"
git push origin main --tags
```

## GitHub Release

1. Create a GitHub Release from tag `v0.2.0`
2. Copy the `## 0.2.0` section from `CHANGELOG.md` (or use the snippet below)

Optional helper to print the `CHANGELOG.md` section:

```bash
python - <<'PY'
from pathlib import Path

text = Path("CHANGELOG.md").read_text(encoding="utf-8").splitlines()
start = None
end = None
for i, line in enumerate(text):
    if line.strip() == "## 0.2.0":
        start = i
        continue
    if start is not None and line.startswith("## ") and i > start:
        end = i
        break
block = text[start:end] if start is not None else []
print("\n".join(block).strip() + "\n")
PY
```

## Suggested Env Vars

Minimum:

```bash
export CAIA_REGISTRY_API_KEY=change-me
export CAIA_REGISTRY_CORS_ORIGINS=""
```

Common:

```bash
export CAIA_REGISTRY_DB_PATH=./registry.db
export CAIA_REGISTRY_SQLITE_BUSY_TIMEOUT_MS=5000
export CAIA_REGISTRY_MAX_HASH_BYTES=268435456
export CAIA_REGISTRY_MAX_METRICS_BODY_BYTES=65536
export CAIA_REGISTRY_MAX_EVENT_PAYLOAD_BYTES=65536
export CAIA_REGISTRY_MAX_WRITE_BODY_BYTES=262144
export CAIA_REGISTRY_TRUST_ACTOR_HEADER=0
```

## Release Notes Snippet (copy/paste)

### Highlights

- Secure-by-default API key auth + locked-down CORS
- SQLite WAL + busy timeout, per-request connections
- Artifact identity (sha256 + size), no checkpoint loading/unpickling
- Audit trail (`model_events`) + suite-scoped eval metrics + optional promotion gating
- CLIs: `ingest.py`, `bench.py`, `maintenance.py`

### QA

Run: `make qa`

- Tests: 38 total, 0 failed (pass_rate 1.0)
- Coverage: 83.78%
- Lint: ruff 0 issues
- Security: pip-audit 0 vulns
- Concurrency: ok (200 total, 0 failed; p50 2.71ms, p95 41.08ms)
