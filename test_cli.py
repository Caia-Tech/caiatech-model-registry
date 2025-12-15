from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


def _load_server(monkeypatch, tmp_path: Path, *, api_key: str = "test-key"):
    server_dir = Path(__file__).resolve().parent
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    monkeypatch.setenv("CAIA_REGISTRY_DB_PATH", str(tmp_path / "cli_registry.db"))
    monkeypatch.setenv("CAIA_REGISTRY_API_KEY", api_key)
    monkeypatch.setenv("CAIA_REGISTRY_CORS_ORIGINS", "")
    monkeypatch.setenv("CAIA_REGISTRY_ALLOW_LOCAL_PATHS", "1")
    monkeypatch.setenv("CAIA_REGISTRY_EXPOSE_LOCAL_PATHS", "0")
    monkeypatch.delenv("CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE", raising=False)

    import server

    return importlib.reload(server)


def test_ingest_sha256_and_size(tmp_path: Path):
    import ingest

    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()

    sha, size = ingest.sha256_and_size(p)
    assert sha == expected
    assert size == 5


def test_ingest_dry_run_builds_payload_and_defaults_artifact_uri(monkeypatch, tmp_path: Path, capsys):
    import ingest

    monkeypatch.setenv("CAIA_REGISTRY_API_KEY", "dev-key")

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"checkpoint-bytes")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"architecture": {"d_model": 64}}), encoding="utf-8")

    rc = ingest.main(
        [
            "--registry-url",
            "http://example",
            "--name",
            "demo",
            "--version",
            "v1",
            "--checkpoint-path",
            str(ckpt),
            "--config-path",
            str(cfg),
            "--tag",
            "a",
            "--tag",
            "b",
            "--dry-run",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["artifact_uri"].startswith("sha256:")
    assert payload["checkpoint_sha256"] == hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert payload["checkpoint_size_bytes"] == ckpt.stat().st_size
    assert payload["config_sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert payload["config_size_bytes"] == cfg.stat().st_size
    assert payload["config_json"] == {"architecture": {"d_model": 64}}
    assert payload["tags"] == ["a", "b"]
    assert "checkpoint_path" not in payload
    assert "config_path" not in payload


def test_ingest_metrics_json_and_metrics_path_are_mutually_exclusive(monkeypatch, tmp_path: Path):
    import ingest

    monkeypatch.setenv("CAIA_REGISTRY_API_KEY", "dev-key")

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        ingest.main(
            [
                "--registry-url",
                "http://example",
                "--name",
                "demo",
                "--version",
                "v1",
                "--artifact-uri",
                "s3://bucket/x",
                "--metrics-json",
                "{}",
                "--metrics-path",
                str(metrics_path),
                "--dry-run",
            ]
        )
    assert "Use only one" in str(e.value)


def test_bench_compute_smoke_metrics_reads_checkpoint_and_config(tmp_path: Path):
    import bench

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"1234567890")
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")

    metrics = bench.compute_smoke_metrics({}, checkpoint_path=str(ckpt), config_path=str(cfg))
    assert metrics["artifact_accessible"] is True
    assert metrics["checkpoint_size_bytes"] == ckpt.stat().st_size
    assert metrics["config_parse_ok"] is True
    assert metrics["score"] == 1.0


def test_bench_dry_run_builds_expected_payload(monkeypatch, tmp_path: Path, capsys):
    import bench

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"xx")
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")

    def fake_load_model(*args, **kwargs):
        return {"id": 123}

    monkeypatch.setattr(bench, "load_model", fake_load_model)

    rc = bench.main(
        [
            "--registry-url",
            "http://example",
            "--api-key",
            "dev-key",
            "--model-id",
            "123",
            "--suite",
            "smoke-v1",
            "--checkpoint-path",
            str(ckpt),
            "--config-path",
            str(cfg),
            "--eval-commit",
            "deadbeef",
            "--notes",
            "hello",
            "--dry-run",
        ]
    )
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["model_id"] == 123
    payload = out["payload"]
    assert payload["suite"] == "smoke-v1"
    assert payload["metrics"]["config_parse_ok"] is True
    assert payload["metrics"]["score"] == 1.0
    assert payload["eval_commit"] == "deadbeef"
    assert payload["notes"] == "hello"
    assert payload["eval_config"]["runner"] == "bench.py"


def test_bench_posts_metrics_merges_suites_and_logs_eval_event(monkeypatch, tmp_path: Path):
    import bench

    server = _load_server(monkeypatch, tmp_path, api_key="test-key")
    with TestClient(server.app) as client:
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={
                "name": "bench-demo",
                "version": "v1",
                "artifact_uri": "s3://bucket/ckpt.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_size_bytes": 1,
            },
        )
        assert r.status_code == 200, r.text
        model_id = r.json()["id"]

        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"1234")
        cfg = tmp_path / "config.json"
        cfg.write_text("{}", encoding="utf-8")

        def http_json_via_client(method: str, url: str, *, api_key: str, body: dict | None = None) -> Any:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            resp = client.request(method, path, headers={"X-API-Key": api_key}, json=body)
            if resp.status_code >= 400:
                raise SystemExit(f"HTTP {resp.status_code}: {resp.text}")
            return resp.json() if resp.text else None

        monkeypatch.setattr(bench, "http_json", http_json_via_client)

        rc1 = bench.main(
            [
                "--registry-url",
                "http://registry",
                "--api-key",
                "test-key",
                "--model-id",
                str(model_id),
                "--suite",
                "smoke-v1",
                "--checkpoint-path",
                str(ckpt),
                "--config-path",
                str(cfg),
            ]
        )
        assert rc1 == 0

        rc2 = bench.main(
            [
                "--registry-url",
                "http://registry",
                "--api-key",
                "test-key",
                "--model-id",
                str(model_id),
                "--suite",
                "core-v1",
                "--checkpoint-path",
                str(ckpt),
                "--config-path",
                str(cfg),
            ]
        )
        assert rc2 == 0

        suites = client.get(f"/models/{model_id}/metrics", headers={"X-API-Key": "test-key"}).json()
        assert "smoke-v1" in suites
        assert "core-v1" in suites

        events = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "test-key"}).json()
        assert any(e["event_type"] == "eval" for e in events), events


def test_concurrency_smoke_script_runs_and_writes_report(tmp_path: Path):
    import concurrency_smoke

    out_path = tmp_path / "concurrency.json"
    rc = concurrency_smoke.main(["--threads", "2", "--requests-per-thread", "3", "--out", str(out_path)])
    assert rc == 0

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["requests_total"] == 6
