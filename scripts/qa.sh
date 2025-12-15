#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${QA_OUT_DIR:-qa_artifacts}"
mkdir -p "$OUT_DIR"

python - <<'PY'
import json
import time
from pathlib import Path

out_dir = Path(__import__("os").environ.get("QA_OUT_DIR", "qa_artifacts"))
out_dir.mkdir(parents=True, exist_ok=True)
meta = {"started_at_epoch": time.time()}
out_dir.joinpath("qa_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

status=0

echo "[qa] py_compile"
python -m py_compile \
  server.py ingest.py bench.py smoke_test.py qa_report.py concurrency_smoke.py maintenance.py \
  scripts/pre_release_audit.py test_server.py test_cli.py test_qa_report.py test_maintenance.py

echo "[qa] smoke_test"
python smoke_test.py

echo "[qa] concurrency"
set +e
python concurrency_smoke.py \
  --threads "${QA_CONCURRENCY_THREADS:-10}" \
  --requests-per-thread "${QA_CONCURRENCY_REQUESTS_PER_THREAD:-20}" \
  --out "$OUT_DIR/concurrency.json" 2>&1 | tee "$OUT_DIR/concurrency.txt"
concurrency_status=${PIPESTATUS[0]}
set -e
if [ "$concurrency_status" -ne 0 ]; then
  status="$concurrency_status"
fi

echo "[qa] pytest durations"
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --durations=10 --junitxml="$OUT_DIR/pytest_results.xml" 2>&1 | tee "$OUT_DIR/pytest_durations.txt"
pytest_status=${PIPESTATUS[0]}
set -e
if [ "$pytest_status" -ne 0 ]; then
  status="$pytest_status"
fi

echo "[qa] pytest coverage"
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov --cov=. --cov-report=term-missing --cov-report="json:$OUT_DIR/coverage.json" --junitxml="$OUT_DIR/pytest_cov_results.xml" 2>&1 | tee "$OUT_DIR/pytest_cov.txt"
pytest_cov_status=${PIPESTATUS[0]}
set -e
if [ "$pytest_cov_status" -ne 0 ]; then
  status="$pytest_cov_status"
fi

echo "[qa] ruff"
if command -v ruff >/dev/null 2>&1; then
  ruff check . --output-format json --exit-zero > "$OUT_DIR/ruff.json" || true
else
  echo '{"error":"ruff not installed"}' > "$OUT_DIR/ruff.json"
  echo "ruff not installed" > "$OUT_DIR/ruff_missing.txt"
fi

echo "[qa] pip-audit"
if command -v pip-audit >/dev/null 2>&1; then
  set +e
  pip-audit -f json -r requirements.txt -r requirements-dev.txt > "$OUT_DIR/pip_audit.json"
  set -e
else
  echo '{"error":"pip-audit not installed"}' > "$OUT_DIR/pip_audit.json"
  echo "pip-audit not installed" > "$OUT_DIR/pip_audit_missing.txt"
fi

python - <<'PY'
import json
import time
from pathlib import Path

out_dir = Path(__import__("os").environ.get("QA_OUT_DIR", "qa_artifacts"))
meta_path = out_dir / "qa_meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
meta["ended_at_epoch"] = time.time()
meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

echo "[qa] qa_report"
python qa_report.py --artifacts "$OUT_DIR" --out "$OUT_DIR/qa_report.json"
cat "$OUT_DIR/qa_report.json"

exit "$status"
