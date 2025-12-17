from __future__ import annotations

import sys
import types
import hashlib
import json
from pathlib import Path

import pytest


def test_parse_checkpoint_filename_extracts_step_and_final_flag():
    import publish_checkpoint as pc

    a = pc.parse_checkpoint_filename("checkpoint_step_500.pt")
    assert a.step == 500
    assert a.is_final is False

    b = pc.parse_checkpoint_filename("checkpoint_final_step_5543.pt")
    assert b.step == 5543
    assert b.is_final is True

    c = pc.parse_checkpoint_filename("weird_name.pt")
    assert c.step is None
    assert c.is_final is False


def test_find_latest_checkpoint_prefers_highest_step(tmp_path: Path):
    import publish_checkpoint as pc

    (tmp_path / "checkpoint_step_1000.pt").write_bytes(b"a")
    (tmp_path / "checkpoint_step_9000.pt").write_bytes(b"b")
    (tmp_path / "checkpoint_final_step_8000.pt").write_bytes(b"c")

    latest = pc.find_latest_checkpoint(tmp_path)
    assert latest.step == 9000
    assert latest.path.name == "checkpoint_step_9000.pt"


def test_dry_run_builds_payload_with_template_and_defaults_version(monkeypatch, tmp_path: Path, capsys):
    import publish_checkpoint as pc

    monkeypatch.setenv("CAIA_REGISTRY_API_KEY", "dev-key")

    ckpt = tmp_path / "checkpoint_step_500.pt"
    ckpt.write_bytes(b"checkpoint-bytes")
    sha = hashlib.sha256(b"checkpoint-bytes").hexdigest()

    rc = pc.main(
        [
            "--registry-url",
            "http://example",
            "--name",
            "onyx",
            "--checkpoint-path",
            str(ckpt),
            "--artifact-uri-template",
            "s3://artifacts/{name}/{basename}",
            "--dry-run",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["name"] == "onyx"
    assert payload["version"] == "step-500"
    assert payload["artifact_uri"] == "s3://artifacts/onyx/checkpoint_step_500.pt"
    assert payload["checkpoint_sha256"] == sha
    assert payload["checkpoint_size_bytes"] == ckpt.stat().st_size
    assert payload["training_step"] == 500


def test_errors_when_no_checkpoint_files(tmp_path: Path):
    import publish_checkpoint as pc

    with pytest.raises(SystemExit) as e:
        pc.find_latest_checkpoint(tmp_path)
    assert "No checkpoint files found" in str(e.value)


def test_upload_to_s3_skips_when_exists_and_size_matches(tmp_path: Path, monkeypatch):
    import publish_checkpoint as pc

    f = tmp_path / "checkpoint_step_1.pt"
    f.write_bytes(b"abc")
    size = f.stat().st_size

    calls = {"upload": 0, "head": 0}

    class S3:
        def head_object(self, Bucket, Key):  # noqa: N802
            _ = (Bucket, Key)
            calls["head"] += 1
            return {"ContentLength": size}

        def upload_file(self, *a, **k):
            calls["upload"] += 1

    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = lambda *a, **k: S3()

    botocore_cfg_mod = types.ModuleType("botocore.config")

    class Config:  # noqa: D401
        def __init__(self, *a, **k):
            _ = (a, k)

    botocore_cfg_mod.Config = Config

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_cfg_mod)

    pc.upload_to_s3(
        local_path=f,
        artifact_uri="s3://artifacts/x.pt",
        size_bytes=size,
        endpoint_url="http://minio:9000",
        region="us-east-1",
        skip_if_exists=True,
        force=False,
    )
    assert calls["upload"] == 0
    assert calls["head"] == 1


def test_upload_to_s3_uploads_and_verifies_size(tmp_path: Path, monkeypatch):
    import publish_checkpoint as pc

    f = tmp_path / "checkpoint_step_1.pt"
    f.write_bytes(b"abc")
    size = f.stat().st_size

    calls = {"upload": 0, "head": 0}
    exists = {"uploaded": False}

    class S3:
        def head_object(self, Bucket, Key):  # noqa: N802
            _ = (Bucket, Key)
            calls["head"] += 1
            if not exists["uploaded"]:
                raise Exception("not found")
            return {"ContentLength": size}

        def upload_file(self, Filename, Bucket, Key):  # noqa: N802
            _ = (Filename, Bucket, Key)
            calls["upload"] += 1
            exists["uploaded"] = True

    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = lambda *a, **k: S3()

    botocore_cfg_mod = types.ModuleType("botocore.config")

    class Config:  # noqa: D401
        def __init__(self, *a, **k):
            _ = (a, k)

    botocore_cfg_mod.Config = Config

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_cfg_mod)

    pc.upload_to_s3(
        local_path=f,
        artifact_uri="s3://artifacts/x.pt",
        size_bytes=size,
        endpoint_url="http://minio:9000",
        region="us-east-1",
        skip_if_exists=True,
        force=False,
    )
    assert calls["upload"] == 1
    assert calls["head"] >= 1
