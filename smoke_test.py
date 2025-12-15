#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


def main() -> int:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.environ["CAIA_REGISTRY_DB_PATH"] = str(tmp_path / "smoke.db")
        os.environ["CAIA_REGISTRY_API_KEY"] = "smoke-key"
        os.environ["CAIA_REGISTRY_CORS_ORIGINS"] = ""
        os.environ["CAIA_REGISTRY_ALLOW_LOCAL_PATHS"] = "1"
        os.environ["CAIA_REGISTRY_EXPOSE_LOCAL_PATHS"] = "0"
        os.environ["CAIA_REGISTRY_PROMOTION_REQUIRED_SUITE"] = "smoke-v1"

        import server

        server = importlib.reload(server)
        with TestClient(server.app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/health").status_code == 200
            assert client.get("/models").status_code == 401

            checkpoint_path = tmp_path / "checkpoint.bin"
            checkpoint_path.write_bytes(b"checkpoint-bytes")

            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({"architecture": {"d_model": 64}}), encoding="utf-8")

            r = client.post(
                "/models",
                headers={"X-API-Key": "smoke-key"},
                json={
                    "name": "smoke",
                    "version": "v1",
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                },
            )
            assert r.status_code == 200, r.text
            model = r.json()
            assert model["artifact_uri"].startswith("sha256:")

            model_id = model["id"]
            events = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "smoke-key"}).json()
            assert any(e["event_type"] == "create" for e in events)

            promote = client.post(
                f"/models/{model_id}/promote",
                headers={"X-API-Key": "smoke-key"},
                params={"to_status": "production"},
            )
            assert promote.status_code == 409

            r = client.post(
                f"/models/{model_id}/metrics",
                headers={"X-API-Key": "smoke-key"},
                json={
                    "suite": "smoke-v1",
                    "metrics": {"score": 1.0, "config_parse_ok": True},
                    "dataset": "smoke",
                    "eval_commit": "deadbeef",
                    "notes": "smoke_test",
                },
            )
            assert r.status_code == 200, r.text
            updated = r.json()
            assert updated["metrics"]["suites"]["smoke-v1"]["score"] == 1.0

            r = client.post(
                f"/models/{model_id}/metrics",
                headers={"X-API-Key": "smoke-key"},
                json={
                    "suite": "core-v1",
                    "metrics": {"score": 0.5},
                },
            )
            assert r.status_code == 200, r.text

            suites = client.get(f"/models/{model_id}/metrics", headers={"X-API-Key": "smoke-key"}).json()
            assert "smoke-v1" in suites
            assert "core-v1" in suites

            events2 = client.get(f"/models/{model_id}/events", headers={"X-API-Key": "smoke-key"}).json()
            assert any(e["event_type"] == "eval" for e in events2)

            promote2 = client.post(
                f"/models/{model_id}/promote",
                headers={"X-API-Key": "smoke-key"},
                params={"to_status": "production"},
            )
            assert promote2.status_code == 200, promote2.text
            assert promote2.json()["status"] == "production"

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
