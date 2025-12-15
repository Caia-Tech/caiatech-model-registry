from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _load_server(
    monkeypatch,
    tmp_path: Path,
    *,
    allow_local_paths: bool = False,
    expose_local_paths: bool = False,
    trust_actor_header: bool = False,
    cors_origins: str | None = "",
    api_key: str | None = "test-key",
    promotion_required_suite: str | None = None,
    max_hash_bytes: int | None = None,
    max_metrics_body_bytes: int | None = None,
    max_event_payload_bytes: int | None = None,
    max_write_body_bytes: int | None = None,
    sqlite_busy_timeout_ms: int | None = None,
):
    server_dir = Path(__file__).resolve().parent
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    monkeypatch.setenv("CAIA_REGISTRY_DB_PATH", str(tmp_path / "test_registry.db"))
    if api_key is None:
        monkeypatch.delenv("CAIA_REGISTRY_API_KEY", raising=False)
        monkeypatch.delenv("CAIA_REGISTRY_API_KEYS", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_API_KEY", api_key)
        monkeypatch.delenv("CAIA_REGISTRY_API_KEYS", raising=False)

    if cors_origins is None:
        monkeypatch.delenv("CAIA_REGISTRY_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_CORS_ORIGINS", cors_origins)

    monkeypatch.setenv("CAIA_REGISTRY_ALLOW_LOCAL_PATHS", "1" if allow_local_paths else "0")
    monkeypatch.setenv("CAIA_REGISTRY_EXPOSE_LOCAL_PATHS", "1" if expose_local_paths else "0")
    monkeypatch.setenv("CAIA_REGISTRY_TRUST_ACTOR_HEADER", "1" if trust_actor_header else "0")
    if promotion_required_suite is None:
        monkeypatch.delenv("CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE", promotion_required_suite)

    if max_hash_bytes is None:
        monkeypatch.delenv("CAIA_REGISTRY_MAX_HASH_BYTES", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_MAX_HASH_BYTES", str(max_hash_bytes))

    if max_metrics_body_bytes is None:
        monkeypatch.delenv("CAIA_REGISTRY_MAX_METRICS_BODY_BYTES", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_MAX_METRICS_BODY_BYTES", str(max_metrics_body_bytes))

    if max_event_payload_bytes is None:
        monkeypatch.delenv("CAIA_REGISTRY_MAX_EVENT_PAYLOAD_BYTES", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_MAX_EVENT_PAYLOAD_BYTES", str(max_event_payload_bytes))

    if max_write_body_bytes is None:
        monkeypatch.delenv("CAIA_REGISTRY_MAX_WRITE_BODY_BYTES", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_MAX_WRITE_BODY_BYTES", str(max_write_body_bytes))

    if sqlite_busy_timeout_ms is None:
        monkeypatch.delenv("CAIA_REGISTRY_SQLITE_BUSY_TIMEOUT_MS", raising=False)
    else:
        monkeypatch.setenv("CAIA_REGISTRY_SQLITE_BUSY_TIMEOUT_MS", str(sqlite_busy_timeout_ms))

    import server

    return importlib.reload(server)


def test_root_and_health_are_public(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200


def test_auth_required_for_models(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert client.get("/models").status_code == 401
        assert client.get("/models", headers={"X-API-Key": "wrong"}).status_code == 401


def test_auth_returns_503_when_unconfigured(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, api_key=None)
    with TestClient(server.app) as client:
        r = client.get("/models")
        assert r.status_code == 503, r.text


def test_actor_header_ignored_by_default(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, trust_actor_header=False)
    with TestClient(server.app) as client:
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key", "X-Actor": "evil"},
            json={"name": "actor", "version": "v1", "artifact_uri": "s3://x", "checkpoint_sha256": "a" * 64},
        )
        assert r.status_code == 200, r.text
        model_id = r.json()["id"]

        events = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "test-key"}).json()
        assert events[0]["event_type"] == "create"
        assert events[0]["actor"] != "evil"
        assert events[0]["actor"].startswith("key:test")


def test_actor_header_trusted_when_enabled(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, trust_actor_header=True)
    with TestClient(server.app) as client:
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key", "X-Actor": "alice"},
            json={"name": "actor2", "version": "v1", "artifact_uri": "s3://x", "checkpoint_sha256": "a" * 64},
        )
        assert r.status_code == 200, r.text
        model_id = r.json()["id"]

        events = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "test-key"}).json()
        assert events[0]["event_type"] == "create"
        assert events[0]["actor"] == "alice"


def test_cors_defaults_localhost_only(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, cors_origins=None)
    with TestClient(server.app) as client:
        allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"

        denied = client.get("/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in denied.headers


def test_cors_empty_allows_none(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, cors_origins="")
    with TestClient(server.app) as client:
        denied = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" not in denied.headers


def test_local_paths_blocked_when_disabled(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False)
    with TestClient(server.app) as client:
        checkpoint_path = tmp_path / "checkpoint.bin"
        checkpoint_path.write_bytes(b"hello")
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "no-local", "version": "v1", "checkpoint_path": str(checkpoint_path)},
        )
        assert r.status_code == 400


def test_expose_local_paths_flag(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=True, expose_local_paths=True)
    with TestClient(server.app) as client:
        checkpoint_path = tmp_path / "checkpoint.bin"
        checkpoint_path.write_bytes(b"hello-checkpoint")
        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "paths", "version": "v1", "checkpoint_path": str(checkpoint_path), "config_path": str(config_path)},
        )
        assert r.status_code == 200, r.text
        model = r.json()
        assert model["local_checkpoint_path"] == str(checkpoint_path)
        assert model["local_config_path"] == str(config_path)


def test_models_post_body_size_limit(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, max_write_body_bytes=200)
    with TestClient(server.app) as client:
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={
                "name": "too-big",
                "version": "v1",
                "artifact_uri": "s3://x",
                "checkpoint_sha256": "a" * 64,
                "description": "x" * 2000,
            },
        )
        assert r.status_code == 413, r.text


def test_models_patch_body_size_limit(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, max_write_body_bytes=200)
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "patch-big", "version": "v1", "artifact_uri": "s3://x", "checkpoint_sha256": "a" * 64},
        )
        assert created.status_code == 200, created.text
        model_id = created.json()["id"]

        r = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"description": "y" * 2000},
        )
        assert r.status_code == 413, r.text


def test_metrics_body_size_limit(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, max_metrics_body_bytes=200)
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "limit", "version": "v1", "artifact_uri": "s3://x", "checkpoint_sha256": "a" * 64},
        ).json()
        model_id = created["id"]

        big = "x" * 1000
        r = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"blob": big}},
        )
        assert r.status_code == 413, r.text


def test_event_payload_size_limit(monkeypatch, tmp_path: Path):
    server = _load_server(
        monkeypatch,
        tmp_path,
        max_metrics_body_bytes=4096,
        max_event_payload_bytes=512,
    )
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "eventlimit", "version": "v1", "artifact_uri": "s3://x", "checkpoint_sha256": "a" * 64},
        ).json()
        model_id = created["id"]

        r = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"blob": "x" * 900}},
        )
        assert r.status_code == 413, r.text


def test_hashing_limit_skips_sha(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=True, max_hash_bytes=1)
    with TestClient(server.app) as client:
        checkpoint_path = tmp_path / "checkpoint.bin"
        checkpoint_path.write_bytes(b"this-is-bigger-than-1-byte")
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={
                "name": "big",
                "version": "v1",
                "artifact_uri": "local://big",
                "checkpoint_path": str(checkpoint_path),
            },
        )
        assert r.status_code == 200, r.text
        model = r.json()
        assert model["checkpoint_sha256"] is None
        assert model["checkpoint_size_bytes"] == checkpoint_path.stat().st_size


def test_sha256_validation(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False)
    with TestClient(server.app) as client:
        r = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={
                "name": "badsha",
                "version": "v1",
                "artifact_uri": "s3://bucket/x",
                "checkpoint_sha256": "123",
            },
        )
        assert r.status_code == 400


def test_register_and_events(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=True, promotion_required_suite=None)
    with TestClient(server.app) as client:
        checkpoint_path = tmp_path / "checkpoint.bin"
        checkpoint_path.write_bytes(b"hello-checkpoint")

        config_path = tmp_path / "config.json"
        config_path.write_text(
            '{"architecture": {"d_model": 123, "n_layers": 4, "n_heads": 2, "vocab_size": 99}}',
            encoding="utf-8",
        )

        resp = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={
                "name": "demo",
                "version": "v1",
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
            },
        )
        assert resp.status_code == 200, resp.text
        created = resp.json()
        assert created["name"] == "demo"
        assert created["version"] == "v1"
        assert created["artifact_uri"].startswith("sha256:")
        assert created["checkpoint_sha256"]
        assert created["checkpoint_size_bytes"] == len(b"hello-checkpoint")
        assert created["d_model"] == 123
        assert created["local_checkpoint_path"] is None
        assert created["local_config_path"] is None

        model_id = created["id"]
        events = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "test-key"}).json()
        assert any(e["event_type"] == "create" for e in events)

        promoted = client.post(
            f"/models/{model_id}/promote",
            headers={"X-API-Key": "test-key"},
            params={"to_status": "production"},
        ).json()
        assert promoted["status"] == "production"

        events2 = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "test-key"}).json()
        assert any(e["event_type"] == "promote" for e in events2)


def test_metrics_write_and_promotion_gate(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=True, promotion_required_suite="smoke-v1")
    with TestClient(server.app) as client:
        checkpoint_path = tmp_path / "checkpoint.bin"
        checkpoint_path.write_bytes(b"hello-checkpoint")

        config_path = tmp_path / "config.json"
        config_path.write_text('{"architecture": {"d_model": 1}}', encoding="utf-8")

        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={
                "name": "gated",
                "version": "v1",
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
            },
        ).json()
        model_id = created["id"]

        promote = client.post(
            f"/models/{model_id}/promote",
            headers={"X-API-Key": "test-key"},
            params={"to_status": "production"},
        )
        assert promote.status_code == 409

        r = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"score": 1.0}},
        )
        assert r.status_code == 200, r.text
        suites = client.get(f"/models/{model_id}/metrics", headers={"X-API-Key": "test-key"}).json()
        assert "smoke-v1" in suites

        promote2 = client.post(
            f"/models/{model_id}/promote",
            headers={"X-API-Key": "test-key"},
            params={"to_status": "production"},
        )
        assert promote2.status_code == 200, promote2.text
        events = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "test-key"}).json()
        assert any(e["event_type"] == "eval" for e in events)


def test_metrics_auth_and_validation(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite=None)
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={
                "name": "metrics",
                "version": "v1",
                "artifact_uri": "s3://bucket/x",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_size_bytes": 1,
            },
        ).json()
        model_id = created["id"]

        assert client.get(f"/models/{model_id}/metrics").status_code == 401
        assert client.post(f"/models/{model_id}/metrics", json={}).status_code == 401

        r = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "   ", "metrics": {"score": 1.0}},
        )
        assert r.status_code == 400

        r2 = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1"},
        )
        assert r2.status_code == 422

        missing_suite = client.get(f"/models/{model_id}/metrics?suite=smoke-v1", headers={"X-API-Key": "test-key"})
        assert missing_suite.status_code == 404


def test_metrics_merge_behavior(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite=None)
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "merge", "version": "v1", "artifact_uri": "s3://bucket/x", "checkpoint_sha256": "a" * 64},
        ).json()
        model_id = created["id"]

        r1 = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"a": 1, "score": 1.0}},
        )
        assert r1.status_code == 200, r1.text

        r2 = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"b": 2}},
        )
        assert r2.status_code == 200, r2.text

        suites = client.get(f"/models/{model_id}/metrics", headers={"X-API-Key": "test-key"}).json()
        assert suites["smoke-v1"]["a"] == 1
        assert suites["smoke-v1"]["b"] == 2
        assert suites["smoke-v1"]["_meta"]["actor"].startswith("key:")


def test_metrics_optimistic_locking_conflict(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite=None)
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "lock", "version": "v1", "artifact_uri": "s3://bucket/x", "checkpoint_sha256": "a" * 64},
        )
        assert created.status_code == 200, created.text
        model_id = created.json()["id"]

        current = client.get(f"/models/{model_id}", headers={"X-API-Key": "test-key"}).json()
        t0 = current["updated_at"]

        ok = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            params={"if_updated_at": t0},
            json={"suite": "smoke-v1", "metrics": {"score": 1.0}},
        )
        assert ok.status_code == 200, ok.text

        conn = sqlite3.connect(tmp_path / "test_registry.db")
        try:
            conn.execute("UPDATE models SET updated_at = '1999-01-01 00:00:00' WHERE id = ?", (model_id,))
            conn.commit()
        finally:
            conn.close()

        conflict = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            params={"if_updated_at": t0},
            json={"suite": "smoke-v1", "metrics": {"x": 1}},
        )
        assert conflict.status_code == 409, conflict.text

        latest = client.get(f"/models/{model_id}", headers={"X-API-Key": "test-key"}).json()
        t1 = latest["updated_at"]
        ok2 = client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key", "If-Updated-At": t1},
            json={"suite": "smoke-v1", "metrics": {"y": 2}},
        )
        assert ok2.status_code == 200, ok2.text


def test_promotion_gate_blocks_score_regression(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite="smoke-v1")
    with TestClient(server.app) as client:
        base_payload = {"name": "scored", "artifact_uri": "s3://bucket/x", "checkpoint_sha256": "a" * 64}

        v1 = client.post("/models", headers={"X-API-Key": "test-key"}, json={**base_payload, "version": "v1"}).json()
        client.post(
            f"/models/{v1['id']}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"score": 1.0}},
        )
        p1 = client.post(
            f"/models/{v1['id']}/promote", headers={"X-API-Key": "test-key"}, params={"to_status": "production"}
        )
        assert p1.status_code == 200, p1.text

        v2 = client.post("/models", headers={"X-API-Key": "test-key"}, json={**base_payload, "version": "v2"}).json()
        client.post(
            f"/models/{v2['id']}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"score": 0.5}},
        )
        p2 = client.post(
            f"/models/{v2['id']}/promote", headers={"X-API-Key": "test-key"}, params={"to_status": "production"}
        )
        assert p2.status_code == 409

        current = client.get("/production/scored", headers={"X-API-Key": "test-key"}).json()
        assert current["version"] == "v1"


def test_promotion_gate_requires_suite_only_when_missing_score(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite="smoke-v1")
    with TestClient(server.app) as client:
        base_payload = {"name": "noscore", "artifact_uri": "s3://bucket/x", "checkpoint_sha256": "a" * 64}

        v1 = client.post("/models", headers={"X-API-Key": "test-key"}, json={**base_payload, "version": "v1"}).json()
        client.post(
            f"/models/{v1['id']}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"score": 1.0}},
        )
        client.post(f"/models/{v1['id']}/promote", headers={"X-API-Key": "test-key"}, params={"to_status": "production"})

        v2 = client.post("/models", headers={"X-API-Key": "test-key"}, json={**base_payload, "version": "v2"}).json()
        client.post(
            f"/models/{v2['id']}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"config_parse_ok": True}},
        )
        p2 = client.post(
            f"/models/{v2['id']}/promote", headers={"X-API-Key": "test-key"}, params={"to_status": "production"}
        )
        assert p2.status_code == 200, p2.text


def test_production_and_frozen_immutability(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite="smoke-v1")
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "immut", "version": "v1", "artifact_uri": "s3://bucket/x", "checkpoint_sha256": "a" * 64},
        ).json()
        model_id = created["id"]

        freeze = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"frozen": True},
        )
        assert freeze.status_code == 200, freeze.text

        blocked = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"artifact_uri": "s3://bucket/y"},
        )
        assert blocked.status_code == 409

        ok = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"description": "allowed"},
        )
        assert ok.status_code == 200

        unfreeze = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"frozen": False},
        )
        assert unfreeze.status_code == 409


def test_production_immutability_blocks_artifact_updates(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite="smoke-v1")
    with TestClient(server.app) as client:
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "prodimmut", "version": "v1", "artifact_uri": "s3://bucket/x", "checkpoint_sha256": "a" * 64},
        ).json()
        model_id = created["id"]

        client.post(
            f"/models/{model_id}/metrics",
            headers={"X-API-Key": "test-key"},
            json={"suite": "smoke-v1", "metrics": {"score": 1.0}},
        )
        client.post(
            f"/models/{model_id}/promote",
            headers={"X-API-Key": "test-key"},
            params={"to_status": "production"},
        )

        blocked = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"run_id": "new-run"},
        )
        assert blocked.status_code == 409

        ok = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"description": "ok"},
        )
        assert ok.status_code == 200


def test_update_rejects_empty_local_paths(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=True, promotion_required_suite=None)
    with TestClient(server.app) as client:
        checkpoint_path = tmp_path / "checkpoint.bin"
        checkpoint_path.write_bytes(b"x")
        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "emptypath", "version": "v1", "checkpoint_path": str(checkpoint_path)},
        ).json()
        model_id = created["id"]

        r = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"checkpoint_path": ""},
        )
        assert r.status_code == 400


def test_list_models_filters_and_pagination(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, allow_local_paths=False, promotion_required_suite=None)
    with TestClient(server.app) as client:
        def reg(name: str, version: str, step: int, tags: list[str], desc: str):
            r = client.post(
                "/models",
                headers={"X-API-Key": "test-key"},
                json={
                    "name": name,
                    "version": version,
                    "artifact_uri": f"s3://bucket/{name}/{version}",
                    "checkpoint_sha256": ("a" * 63) + str(step % 10),
                    "training_step": step,
                    "tags": tags,
                    "description": desc,
                },
            )
            assert r.status_code == 200, r.text
            return r.json()

        reg("alpha", "v1", 1, ["team-a", "smoke"], "hello world")
        reg("alpha", "v2", 2, ["team-a"], "other")
        reg("beta", "v1", 3, ["team-b"], "hello beta")

        r = client.get(
            "/models",
            headers={"X-API-Key": "test-key"},
            params={"sort": "training_step", "order": "desc"},
        )
        steps = [m["training_step"] for m in r.json()]
        assert steps[:3] == [3, 2, 1]

        r2 = client.get("/models", headers={"X-API-Key": "test-key"}, params={"q": "hello"})
        assert {m["name"] for m in r2.json()} == {"alpha", "beta"}

        r3 = client.get("/models", headers={"X-API-Key": "test-key"}, params={"tag": "smoke"})
        assert len(r3.json()) == 1
        assert r3.json()[0]["name"] == "alpha"

        r4 = client.get("/models", headers={"X-API-Key": "test-key"}, params={"limit": 1, "offset": 1})
        assert len(r4.json()) == 1

        r5 = client.get("/models", headers={"X-API-Key": "test-key"}, params={"limit": 201})
        assert r5.status_code == 422


def test_sqlite_pragmas_and_trigger(monkeypatch, tmp_path: Path):
    server = _load_server(monkeypatch, tmp_path, sqlite_busy_timeout_ms=1234, promotion_required_suite=None)
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200

        conn = server._connect_db()
        try:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0].lower()
            assert mode == "wal"
            busy = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
            assert busy == 1234
        finally:
            conn.close()

        created = client.post(
            "/models",
            headers={"X-API-Key": "test-key"},
            json={"name": "trg", "version": "v1", "artifact_uri": "s3://bucket/x", "checkpoint_sha256": "a" * 64},
        ).json()
        model_id = created["id"]

        conn = sqlite3.connect(tmp_path / "test_registry.db")
        try:
            conn.execute("UPDATE models SET updated_at = '2000-01-01 00:00:00' WHERE id = ?", (model_id,))
            conn.commit()
        finally:
            conn.close()

        updated = client.patch(
            f"/models/{model_id}",
            headers={"X-API-Key": "test-key"},
            json={"description": "bump"},
        ).json()
        assert updated["updated_at"] != "2000-01-01 00:00:00"


def test_schema_migration_from_legacy_models_table(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "test_registry.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT DEFAULT 'experimental',
                checkpoint_path TEXT NOT NULL,
                config_path TEXT,
                d_model INTEGER,
                n_layers INTEGER,
                n_heads INTEGER,
                vocab_size INTEGER,
                params INTEGER,
                training_step INTEGER,
                training_loss REAL,
                dataset TEXT,
                description TEXT,
                tags TEXT,
                metrics TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, version)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    server = _load_server(monkeypatch, tmp_path, promotion_required_suite=None)
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(models)").fetchall()}
        assert "artifact_uri" in cols
        assert "checkpoint_sha256" in cols
        assert "run_id" in cols
        assert "frozen" in cols
        triggers = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()]
        assert "trg_models_set_updated_at" in triggers
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "model_events" in tables
    finally:
        conn.close()
