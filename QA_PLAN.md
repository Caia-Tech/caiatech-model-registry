# QA Plan (Repeatable Checklist)

This checklist is designed to be run locally and in CI via `make qa`.

## Security Defaults

- Auth
  - Verify all endpoints except `/` and `/health` require API key auth.
  - Verify missing API key returns `401` and invalid key returns `401`.
  - Verify when no keys are configured (`CAIA_REGISTRY_API_KEY(S)` unset/empty) protected endpoints return `503`.
  - Verify `X-Actor` is ignored by default (prevents audit spoofing); only trusted when `CAIA_REGISTRY_TRUST_ACTOR_HEADER=1`.
- CORS
  - Default behavior: when `CAIA_REGISTRY_CORS_ORIGINS` is unset, only localhost origins are allowed.
  - When `CAIA_REGISTRY_CORS_ORIGINS` is set to an empty string, no origins are allowed.
  - Ensure responses do not reflect arbitrary `Origin` headers.
- Local path exposure (privacy)
  - Verify `CAIA_REGISTRY_EXPOSE_LOCAL_PATHS=0` keeps `local_checkpoint_path` and `local_config_path` out of API responses.
  - Verify `CAIA_REGISTRY_ALLOW_LOCAL_PATHS=0` blocks registering/updating via local paths.

## Schema + Migrations (SQLite)

- Fresh DB init creates:
  - `models` table with v2 columns (artifact identity, provenance, frozen flag).
  - `model_events` table.
  - Indexes for `name`, `status`, `updated_at`, `run_id`.
  - Trigger to keep `updated_at` current on row updates.
- Migration path:
  - Start from an “old” models table and verify v2 columns are added via `ALTER TABLE`.

## Immutability Rules

- Production and frozen models:
  - When `status == production` OR `frozen == 1`, block edits to artifact/provenance/arch fields.
  - Allow metadata updates (e.g. `description`, `tags`, `metrics`) even when immutable.
  - Verify a frozen model cannot be unfrozen.

## Artifact Identity + Hashing

- Verify server never unpickles / loads checkpoints.
- Hashing limits:
  - When local artifact is larger than `CAIA_REGISTRY_MAX_HASH_BYTES`, server records size but skips hashing (sha256 may be null unless provided by client).
  - Verify hash validation rejects invalid sha256 strings.

## Audit Trail (model_events)

- Verify every create/update/delete/promote writes a `model_events` row.
- Verify metrics writes (`POST /models/{id}/metrics`) write `event_type="eval"` with useful payload (suite, metrics, prior hash, metadata).

## Metrics + Benchmarking

- Metrics write:
  - Verify suite-scoped merge behavior: add/merge suite without wiping other suites.
  - Verify repeat writes to same suite merge keys.
  - Verify empty suite rejected and non-object metrics rejected.
  - Verify optional optimistic locking rejects stale writes with `409` (`if_updated_at` query or `If-Updated-At` header).
  - Verify oversized `POST /models/{id}/metrics` is rejected with `413` (see `CAIA_REGISTRY_MAX_METRICS_BODY_BYTES`).
  - Verify oversized `model_events.payload_json` is rejected with `413` (see `CAIA_REGISTRY_MAX_EVENT_PAYLOAD_BYTES`).
- Metrics read:
  - Verify `GET /models/{id}/metrics` returns suite map.
  - Verify `GET /models/{id}/metrics?suite=...` returns only that suite and 404s for missing suite.

## Promotion Gating

- When `CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE` is set:
  - Promotion to production fails if suite is missing.
  - If an existing production model exists and both have numeric `score`, block promotion if candidate score < production score.
  - If `score` is missing/non-numeric, only require suite existence.

## Listing + Query Behavior

- Pagination: `limit` default 50, max 200; `offset` works.
- Sorting: `created_at|updated_at|training_step` with `asc|desc`.
- Search:
  - `q` fuzzy search over name/version/description.
  - `tag` filter matches JSON tags (naive contains).

## SQLite Concurrency Behavior

- Verify connections enable WAL mode and a non-zero busy timeout.
- Verify server uses one connection per request (behavioral check; no shared global connection).

## Interpreting `qa_report.json`

`make qa` writes a single summary file at `qa_artifacts/qa_report.json` plus raw artifacts.

- `tests_total`, `tests_failed`, `pass_rate`: overall test health (should be `tests_failed == 0`).
- `coverage_percent`: line coverage from `pytest-cov` (aim to keep this rising over time).
- `slow_tests`: top slowest pytest nodeids (useful for keeping CI fast).
- `ruff_issues_total`: lint count (0 is ideal; ruff is run with `--exit-zero` so lint does not fail QA).
- `vulns_found`: dependency audit count (may be `null` if `pip-audit` is missing or cannot run).
- `concurrency_ok`: whether the concurrency smoke test saw any failures/locks (should be `true`).
  - `concurrency_requests_failed` and `concurrency_locked_errors` should be `0`.

Notes:
- If running pytest directly on macOS and you see unrelated plugin errors, run via `make test`/`make qa` (they set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).
