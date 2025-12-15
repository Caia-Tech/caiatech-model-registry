#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def _latency_stats(latencies_ms: list[float]) -> dict[str, Any]:
    if not latencies_ms:
        return {"count": 0}
    latencies_ms_sorted = sorted(latencies_ms)
    return {
        "count": len(latencies_ms_sorted),
        "min": min(latencies_ms_sorted),
        "max": max(latencies_ms_sorted),
        "mean": statistics.mean(latencies_ms_sorted),
        "p50": _percentile(latencies_ms_sorted, 50),
        "p95": _percentile(latencies_ms_sorted, 95),
    }


def run_metrics_concurrency(
    client: TestClient,
    *,
    model_id: int,
    api_key: str,
    suite: str,
    threads: int,
    requests_per_thread: int,
) -> dict[str, Any]:
    lock = threading.Lock()
    latencies_ms: list[float] = []
    status_code_counts: Counter[int] = Counter()
    exception_type_counts: Counter[str] = Counter()
    locked_errors = 0
    failures = 0
    errors_sample: list[str] = []

    def record_error(msg: str) -> None:
        nonlocal failures
        failures += 1
        if len(errors_sample) < 10:
            errors_sample.append(msg)

    def worker(ti: int) -> None:
        nonlocal locked_errors
        nonlocal failures
        for i in range(requests_per_thread):
            payload = {
                "suite": suite,
                "metrics": {"score": 1.0, "thread": ti, "i": i},
                "dataset": "concurrency",
                "notes": f"t{ti}-i{i}",
            }
            start = time.perf_counter()
            try:
                resp = client.post(
                    f"/models/{model_id}/metrics",
                    headers={"X-API-Key": api_key},
                    json=payload,
                )
                dur_ms = (time.perf_counter() - start) * 1000.0
                with lock:
                    latencies_ms.append(dur_ms)
                    status_code_counts[resp.status_code] += 1
                    if resp.status_code != 200:
                        record_error(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:  # noqa: BLE001
                dur_ms = (time.perf_counter() - start) * 1000.0
                msg = f"{type(e).__name__}: {e}"
                with lock:
                    latencies_ms.append(dur_ms)
                    exception_type_counts[type(e).__name__] += 1
                    if "database is locked" in msg.lower():
                        locked_errors += 1
                    record_error(msg[:200])

    threads_list = [threading.Thread(target=worker, args=(ti,)) for ti in range(threads)]
    for t in threads_list:
        t.start()
    for t in threads_list:
        t.join()

    total = threads * requests_per_thread
    success = total - failures
    ok = failures == 0 and locked_errors == 0 and status_code_counts.get(200, 0) == total
    return {
        "ok": ok,
        "requests_total": total,
        "requests_success": success,
        "requests_failed": failures,
        "locked_errors": locked_errors,
        "status_code_counts": dict(status_code_counts),
        "exception_type_counts": dict(exception_type_counts),
        "latency_ms": _latency_stats(latencies_ms),
        "errors_sample": errors_sample,
    }


def _register_seed_model(client: TestClient, *, api_key: str) -> int:
    r = client.post(
        "/models",
        headers={"X-API-Key": api_key},
        json={
            "name": "concurrency-model",
            "version": "v1",
            "artifact_uri": "s3://example/concurrency.pt",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_size_bytes": 1,
        },
    )
    if r.status_code != 200:
        raise SystemExit(f"Failed to register model: HTTP {r.status_code}: {r.text}")
    return int(r.json()["id"])


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Concurrency smoke test for Caia Model Registry (SQLite WAL)")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--requests-per-thread", type=int, default=20)
    p.add_argument("--suite", default="smoke-v1")
    p.add_argument("--out", default="qa_artifacts/concurrency.json")
    args = p.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result: dict[str, Any] = {}

    try:
        here = Path(__file__).resolve().parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            os.environ["CAIA_REGISTRY_DB_PATH"] = str(tmp_path / "concurrency.db")
            os.environ["CAIA_REGISTRY_API_KEY"] = "concurrency-key"
            os.environ["CAIA_REGISTRY_CORS_ORIGINS"] = ""
            os.environ["CAIA_REGISTRY_ALLOW_LOCAL_PATHS"] = "0"
            os.environ["CAIA_REGISTRY_EXPOSE_LOCAL_PATHS"] = "0"

            import server

            server = importlib.reload(server)

            with TestClient(server.app) as client:
                model_id = _register_seed_model(client, api_key="concurrency-key")
                result = run_metrics_concurrency(
                    client,
                    model_id=model_id,
                    api_key="concurrency-key",
                    suite=args.suite,
                    threads=args.threads,
                    requests_per_thread=args.requests_per_thread,
                )
                result["model_id"] = model_id
    except Exception as e:  # noqa: BLE001
        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

    ended = time.time()
    result["started_at_epoch"] = started
    result["ended_at_epoch"] = ended
    result["runtime_seconds"] = max(ended - started, 0.0)

    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

