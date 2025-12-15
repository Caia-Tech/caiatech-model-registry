# Contributing

Thanks for taking the time to contribute.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
```

## Running Locally

```bash
export CAIA_REGISTRY_API_KEY=dev
uvicorn server:app --host 127.0.0.1 --port 8001 --reload
```

## Tests

Fast checks:

```bash
python -m py_compile server.py bench.py ingest.py smoke_test.py
python smoke_test.py
```

Pytest (recommended):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q test_server.py
```

Full QA (coverage + lint + optional vuln scan):

```bash
make qa
```

## Code Style

- Keep the project dependency-light (stdlib + FastAPI + Pydantic).
- Prefer small, readable changes over abstractions.
- Avoid adding large frameworks (e.g., ORMs) unless clearly justified.

## Pull Requests

- Include a short description and motivation.
- Add or update tests/smoke checks when behavior changes.
- Avoid committing secrets, local databases, or checkpoints.
