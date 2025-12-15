from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> Any:
    try:
        return read_json(path)
    except Exception:
        return None


def parse_junit_xml(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    suites = []
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = [s for s in root.iter("testsuite")]

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    for suite in suites:
        totals["tests"] += int(suite.attrib.get("tests", 0))
        totals["failures"] += int(suite.attrib.get("failures", 0))
        totals["errors"] += int(suite.attrib.get("errors", 0))
        totals["skipped"] += int(suite.attrib.get("skipped", 0))
        try:
            totals["time"] += float(suite.attrib.get("time", 0.0))
        except ValueError:
            pass
    return totals


def parse_pytest_durations(path: Path, *, limit: int = 10) -> list[dict]:
    slow: list[dict] = []
    if not path.exists():
        return slow

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or not line.endswith(".py") and "::" not in line:
            continue
        if "s " not in line and not line.startswith("0."):
            continue
        parts = line.split()
        if not parts:
            continue
        dur_raw = parts[0]
        if not dur_raw.endswith("s"):
            continue
        try:
            duration = float(dur_raw[:-1])
        except ValueError:
            continue
        nodeid = parts[-1]
        if "::" not in nodeid:
            continue
        slow.append({"nodeid": nodeid, "duration_seconds": duration})

    slow.sort(key=lambda x: x["duration_seconds"], reverse=True)
    return slow[:limit]


def parse_coverage_json(path: Path) -> Optional[float]:
    data = safe_read_json(path)
    if not isinstance(data, dict):
        return None
    totals = data.get("totals")
    if not isinstance(totals, dict):
        return None
    pct = totals.get("percent_covered")
    if isinstance(pct, (int, float)):
        return float(pct)
    pct_display = totals.get("percent_covered_display")
    if isinstance(pct_display, str):
        try:
            return float(pct_display)
        except ValueError:
            return None
    return None


def parse_ruff_json(path: Path) -> dict:
    issues = safe_read_json(path)
    if not isinstance(issues, list):
        return {"issues_total": None, "issues_by_code": {}}
    codes = Counter()
    for item in issues:
        if isinstance(item, dict) and isinstance(item.get("code"), str):
            codes[item["code"]] += 1
    return {"issues_total": len(issues), "issues_by_code": dict(codes)}


def parse_pip_audit_json(path: Path) -> dict:
    data = safe_read_json(path)
    deps = None
    if isinstance(data, list):
        deps = data
    elif isinstance(data, dict) and isinstance(data.get("dependencies"), list):
        deps = data["dependencies"]
    if deps is None:
        return {"vulns_total": None, "deps_with_vulns": None}
    vulns_total = 0
    deps_with_vulns = 0
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        vulns = dep.get("vulns")
        if isinstance(vulns, list) and vulns:
            deps_with_vulns += 1
            vulns_total += len(vulns)
    return {"vulns_total": vulns_total, "deps_with_vulns": deps_with_vulns}


def parse_qa_meta(path: Path) -> dict:
    meta = safe_read_json(path)
    if not isinstance(meta, dict):
        return {}
    started = meta.get("started_at_epoch")
    ended = meta.get("ended_at_epoch")
    runtime = None
    if isinstance(started, (int, float)) and isinstance(ended, (int, float)) and ended >= started:
        runtime = float(ended - started)
    return {"started_at_epoch": started, "ended_at_epoch": ended, "qa_runtime_seconds": runtime}


def parse_concurrency_json(path: Path) -> dict:
    data = safe_read_json(path)
    if not isinstance(data, dict):
        return {"concurrency_ok": None}

    latency = data.get("latency_ms") if isinstance(data.get("latency_ms"), dict) else {}
    return {
        "concurrency_ok": data.get("ok") if isinstance(data.get("ok"), bool) else None,
        "concurrency_requests_total": data.get("requests_total") if isinstance(data.get("requests_total"), int) else None,
        "concurrency_requests_success": data.get("requests_success")
        if isinstance(data.get("requests_success"), int)
        else None,
        "concurrency_requests_failed": data.get("requests_failed") if isinstance(data.get("requests_failed"), int) else None,
        "concurrency_locked_errors": data.get("locked_errors") if isinstance(data.get("locked_errors"), int) else None,
        "concurrency_latency_ms_p50": latency.get("p50") if isinstance(latency.get("p50"), (int, float)) else None,
        "concurrency_latency_ms_p95": latency.get("p95") if isinstance(latency.get("p95"), (int, float)) else None,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Parse QA artifacts and emit qa_report.json")
    p.add_argument("--artifacts", default="qa_artifacts", help="Directory created by scripts/qa.sh")
    p.add_argument("--out", default=None, help="Output JSON path (default: <artifacts>/qa_report.json)")
    args = p.parse_args(argv)

    artifacts = Path(args.artifacts)
    out_path = Path(args.out) if args.out else artifacts / "qa_report.json"

    junit_path = artifacts / "pytest_results.xml"
    junit_totals = parse_junit_xml(junit_path) if junit_path.exists() else {}

    failed = int(junit_totals.get("failures", 0)) + int(junit_totals.get("errors", 0))
    total = int(junit_totals.get("tests", 0))
    skipped = int(junit_totals.get("skipped", 0))
    passed = max(total - failed - skipped, 0)
    pass_rate = (passed / total) if total else None

    slow_tests = parse_pytest_durations(artifacts / "pytest_durations.txt", limit=10)
    coverage_percent = parse_coverage_json(artifacts / "coverage.json")
    ruff_metrics = parse_ruff_json(artifacts / "ruff.json")
    audit_metrics = parse_pip_audit_json(artifacts / "pip_audit.json")
    meta_metrics = parse_qa_meta(artifacts / "qa_meta.json")
    concurrency_metrics = parse_concurrency_json(artifacts / "concurrency.json")

    report = {
        "generated_at_epoch": time.time(),
        "platform": {"python": sys.version.split()[0], "system": platform.system(), "release": platform.release()},
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_skipped": skipped,
        "pass_rate": pass_rate,
        "runtime_seconds": float(junit_totals.get("time", 0.0)) if junit_totals else None,
        "coverage_percent": coverage_percent,
        "slow_tests": slow_tests,
        "ruff_issues_total": ruff_metrics.get("issues_total"),
        "ruff_issues_by_code": ruff_metrics.get("issues_by_code"),
        "vulns_found": audit_metrics.get("vulns_total"),
        "vuln_deps_affected": audit_metrics.get("deps_with_vulns"),
        **meta_metrics,
        **concurrency_metrics,
    }

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
