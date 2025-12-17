.PHONY: dev compile smoke test qa

PY ?= python
HOST ?= 127.0.0.1
PORT ?= 8001

dev:
	CAIA_REGISTRY_API_KEY=$${CAIA_REGISTRY_API_KEY:-dev} uvicorn server:app --host $(HOST) --port $(PORT) --reload

compile:
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m py_compile \
	  server.py bench.py ingest.py publish_checkpoint.py watch_checkpoints.py smoke_test.py qa_report.py concurrency_smoke.py maintenance.py \
	  scripts/pre_release_audit.py test_server.py test_cli.py test_qa_report.py test_maintenance.py

smoke:
	$(PY) smoke_test.py

test: compile smoke
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q

qa:
	bash scripts/qa.sh
