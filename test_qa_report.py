from __future__ import annotations

import json
from pathlib import Path


def test_qa_report_parsers(tmp_path: Path):
    import qa_report

    junit = tmp_path / "pytest_results.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="a" tests="3" failures="1" errors="0" skipped="1" time="0.5" />
  <testsuite name="b" tests="2" failures="0" errors="1" skipped="0" time="1.2" />
</testsuites>
""",
        encoding="utf-8",
    )
    totals = qa_report.parse_junit_xml(junit)
    assert totals == {"tests": 5, "failures": 1, "errors": 1, "skipped": 1, "time": 1.7}

    durations = tmp_path / "pytest_durations.txt"
    durations.write_text(
        "\n".join(
            [
                "============================= slowest 10 durations =============================",
                "0.10s call     test_a.py::test_x",
                "0.05s call     test_b.py::test_y",
                "random noise",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    slow = qa_report.parse_pytest_durations(durations, limit=10)
    assert slow[0]["nodeid"] == "test_a.py::test_x"
    assert slow[0]["duration_seconds"] == 0.10

    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps({"totals": {"percent_covered": 12.34}}), encoding="utf-8")
    assert qa_report.parse_coverage_json(coverage) == 12.34

    ruff = tmp_path / "ruff.json"
    ruff.write_text(
        json.dumps(
            [
                {"code": "F401", "message": "unused import"},
                {"code": "F401", "message": "unused import"},
                {"code": "E501", "message": "line too long"},
            ]
        ),
        encoding="utf-8",
    )
    ruff_metrics = qa_report.parse_ruff_json(ruff)
    assert ruff_metrics["issues_total"] == 3
    assert ruff_metrics["issues_by_code"]["F401"] == 2
    assert ruff_metrics["issues_by_code"]["E501"] == 1

    audit = tmp_path / "pip_audit.json"
    audit.write_text(
        json.dumps(
            {
                "dependencies": [
                    {"name": "a", "vulns": []},
                    {"name": "b", "vulns": [{"id": "X"}, {"id": "Y"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    audit_metrics = qa_report.parse_pip_audit_json(audit)
    assert audit_metrics["vulns_total"] == 2
    assert audit_metrics["deps_with_vulns"] == 1

    concurrency = tmp_path / "concurrency.json"
    concurrency.write_text(
        json.dumps(
            {
                "ok": True,
                "requests_total": 10,
                "requests_success": 10,
                "requests_failed": 0,
                "locked_errors": 0,
                "latency_ms": {"p50": 1.0, "p95": 5.0},
            }
        ),
        encoding="utf-8",
    )
    concurrency_metrics = qa_report.parse_concurrency_json(concurrency)
    assert concurrency_metrics["concurrency_ok"] is True
    assert concurrency_metrics["concurrency_requests_total"] == 10
    assert concurrency_metrics["concurrency_latency_ms_p95"] == 5.0


def test_qa_report_main_writes_summary(tmp_path: Path):
    import qa_report

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    artifacts.joinpath("pytest_results.xml").write_text(
        """<testsuite tests="1" failures="0" errors="0" skipped="0" time="0.1"></testsuite>\n""",
        encoding="utf-8",
    )
    artifacts.joinpath("pytest_durations.txt").write_text("0.01s call     test_a.py::test_x\n", encoding="utf-8")
    artifacts.joinpath("coverage.json").write_text(json.dumps({"totals": {"percent_covered": 99.0}}), encoding="utf-8")
    artifacts.joinpath("ruff.json").write_text("[]", encoding="utf-8")
    artifacts.joinpath("pip_audit.json").write_text(json.dumps({"dependencies": []}), encoding="utf-8")
    artifacts.joinpath("qa_meta.json").write_text(json.dumps({"started_at_epoch": 1.0, "ended_at_epoch": 2.0}), encoding="utf-8")
    artifacts.joinpath("concurrency.json").write_text(json.dumps({"ok": True, "requests_total": 1}), encoding="utf-8")

    out_path = artifacts / "qa_report.json"
    assert qa_report.main(["--artifacts", str(artifacts), "--out", str(out_path)]) == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["tests_total"] == 1
    assert report["tests_failed"] == 0
    assert report["coverage_percent"] == 99.0
    assert report["concurrency_ok"] is True

