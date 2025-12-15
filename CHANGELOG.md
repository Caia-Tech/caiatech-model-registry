# Changelog

All notable changes to this project will be documented in this file.

## 0.2.0

- Secure-by-default API key auth (all endpoints except `/` and `/health`)
- Audit hardening: `X-Actor` ignored unless `CAIA_REGISTRY_TRUST_ACTOR_HEADER=1`
- Locked-down CORS via `CAIA_REGISTRY_CORS_ORIGINS` (localhost-only by default)
- SQLite concurrency improvements (WAL + busy timeout; one connection per request)
- Artifact identity support (sha256 + size; server hashes bytes only, never unpickles)
- Audit trail via `model_events` (create/update/delete/promote/eval)
- Suite-scoped benchmark metrics with `/models/{id}/metrics`
- Payload caps: 64KB limits for metrics requests and event payloads
- Write endpoint body caps for `POST /models` and `PATCH /models/{id}`
- Optional optimistic locking for metrics (`if_updated_at` / `If-Updated-At`)
- Optional promotion gating via `CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE`
- Helper CLIs: `ingest.py` and `bench.py`
- DB maintenance CLI: `maintenance.py` (backup + VACUUM)
- QA: concurrency smoke + CLI tests, and `qa_report.json` includes concurrency metrics
