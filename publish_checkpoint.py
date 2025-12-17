#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_STEP_RE = re.compile(r"checkpoint_(final_)?step_(\d+)\.pt$")


def sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: Optional[int]
    is_final: bool


def parse_checkpoint_filename(name: str) -> CheckpointInfo:
    m = _STEP_RE.search(name)
    if not m:
        return CheckpointInfo(path=Path(name), step=None, is_final=False)
    is_final = bool(m.group(1))
    step = int(m.group(2))
    return CheckpointInfo(path=Path(name), step=step, is_final=is_final)


def find_latest_checkpoint(checkpoint_dir: Path) -> CheckpointInfo:
    candidates = []
    for p in checkpoint_dir.glob("checkpoint_*step_*.pt"):
        info = parse_checkpoint_filename(p.name)
        candidates.append(CheckpointInfo(path=p, step=info.step, is_final=info.is_final))
    if not candidates:
        raise SystemExit(f"No checkpoint files found in: {checkpoint_dir}")

    def key(ci: CheckpointInfo):
        # Prefer known steps; within same step, prefer final.
        step_key = ci.step if ci.step is not None else -1
        final_key = 1 if ci.is_final else 0
        # Tie-break by mtime to make it stable if step parsing fails.
        mtime = ci.path.stat().st_mtime
        return (step_key, final_key, mtime)

    return max(candidates, key=key)


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def parse_s3_uri(uri: str) -> S3Uri:
    s = (uri or "").strip()
    if not s.startswith("s3://"):
        raise ValueError("not s3")
    rest = s[len("s3://") :]
    parts = rest.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("invalid s3 uri")
    return S3Uri(bucket=parts[0], key=parts[1])


def _env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def upload_to_s3(
    *,
    local_path: Path,
    artifact_uri: str,
    size_bytes: int,
    endpoint_url: Optional[str],
    region: str,
    skip_if_exists: bool,
    force: bool,
) -> None:
    """Upload a local file to S3/MinIO at artifact_uri.

    Uses boto3 if available (lazy import). Verifies size via HEAD after upload.
    """
    try:
        import boto3  # type: ignore
        from botocore.config import Config as BotoConfig  # type: ignore
    except Exception as e:
        raise SystemExit("boto3 is required for --upload (pip install boto3)") from e

    parsed = parse_s3_uri(artifact_uri)
    cfg = BotoConfig(
        region_name=region,
        retries={"max_attempts": 5, "mode": "standard"},
        s3={"addressing_style": "path"},
    )
    s3 = boto3.client("s3", endpoint_url=endpoint_url, config=cfg)

    if skip_if_exists and not force:
        try:
            head = s3.head_object(Bucket=parsed.bucket, Key=parsed.key)
            remote_size = int(head.get("ContentLength", -1))
            if remote_size == int(size_bytes):
                return
            raise SystemExit("Remote object exists but size differs; rerun with --force-upload")
        except Exception:
            # Not found or not accessible; proceed to upload.
            pass

    try:
        s3.upload_file(str(local_path), parsed.bucket, parsed.key)
    except Exception as e:
        raise SystemExit(f"Failed to upload to {artifact_uri}") from e

    try:
        head = s3.head_object(Bucket=parsed.bucket, Key=parsed.key)
        remote_size = int(head.get("ContentLength", -1))
    except Exception as e:
        raise SystemExit("Upload finished but could not verify remote object via HEAD") from e

    if remote_size != int(size_bytes):
        raise SystemExit(f"Remote size mismatch after upload: expected {size_bytes}, got {remote_size}")


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    if os.getenv("CAIA_REGISTRY_API_KEY"):
        return os.environ["CAIA_REGISTRY_API_KEY"].strip()
    keys = os.getenv("CAIA_REGISTRY_API_KEYS", "")
    for k in keys.split(","):
        k = k.strip()
        if k:
            return k
    raise SystemExit("Missing API key: pass --api-key or set CAIA_REGISTRY_API_KEY(S)")


def post_json(url: str, api_key: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {detail}") from e


def build_payload(args: argparse.Namespace, *, ckpt: CheckpointInfo) -> dict:
    ckpt_path = ckpt.path
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt_sha, ckpt_size = sha256_and_size(ckpt_path)

    if args.version:
        version = args.version
    else:
        if ckpt.step is not None:
            version = f"{'final-' if ckpt.is_final else ''}step-{ckpt.step}"
        else:
            version = ckpt_path.stem

    artifact_uri = (args.artifact_uri or "").strip() or None
    if not artifact_uri and args.artifact_uri_template:
        artifact_uri = args.artifact_uri_template.format(
            name=args.name,
            version=version,
            step=(ckpt.step if ckpt.step is not None else 0),
            basename=ckpt_path.name,
            sha256=ckpt_sha,
        )
    if not artifact_uri:
        artifact_uri = f"sha256:{ckpt_sha}"

    payload: dict = {
        "name": args.name,
        "version": version,
        "status": args.status,
        "artifact_uri": artifact_uri,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_size_bytes": ckpt_size,
        "run_id": args.run_id,
        "git_commit": args.git_commit,
        "created_by": args.created_by or os.getenv("USER") or None,
        "source_host": args.source_host or socket.gethostname(),
        "training_step": args.training_step if args.training_step is not None else ckpt.step,
        "training_loss": args.training_loss,
        "dataset": args.dataset,
        "description": args.description,
        "tags": args.tag,
    }
    return {k: v for k, v in payload.items() if v is not None}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Publish latest Onyx checkpoint to Caia Model Registry")
    p.add_argument("--registry-url", default="http://localhost:8001", help="Base URL, e.g. http://localhost:8001")
    p.add_argument("--api-key", default=None, help="API key (or set CAIA_REGISTRY_API_KEY)")

    p.add_argument("--name", required=True, help="Model name, e.g. onyx")
    p.add_argument("--version", default=None, help="Registry version label (default derived from checkpoint step)")
    p.add_argument("--status", default="experimental", choices=["experimental", "staging", "production", "archived"])

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint-path", default=None, help="Path to a checkpoint file")
    src.add_argument("--checkpoint-dir", default=None, help="Directory containing checkpoint_step_*.pt files")

    p.add_argument(
        "--artifact-uri",
        default=None,
        help="Artifact URI stored in registry (recommended: s3://...); if omitted defaults to sha256:<checkpoint_sha256>.",
    )
    p.add_argument(
        "--artifact-uri-template",
        default=None,
        help="Format template for artifact URI (e.g. s3://artifacts/{name}/{basename}); supports {name},{version},{step},{basename},{sha256}.",
    )
    p.add_argument("--upload", action="store_true", help="Upload checkpoint to s3://... artifact_uri before registering")
    p.add_argument("--s3-endpoint-url", default=None, help="S3-compatible endpoint (e.g. http://127.0.0.1:9000 for MinIO)")
    p.add_argument("--s3-region", default=None, help="S3 region name (default: AWS_REGION or us-east-1)")
    p.add_argument("--skip-upload-if-exists", action="store_true", help="Skip upload if remote size matches")
    p.add_argument("--force-upload", action="store_true", help="Overwrite remote object if it exists")

    p.add_argument("--run-id", default=None)
    p.add_argument("--git-commit", default=None)
    p.add_argument("--created-by", default=None)
    p.add_argument("--source-host", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--tag", action="append", default=None, help="Repeatable, e.g. --tag onyx --tag eval")

    p.add_argument("--training-step", type=int, default=None)
    p.add_argument("--training-loss", type=float, default=None)

    p.add_argument("--write-metadata", default=None, help="Write the final JSON payload to this path")
    p.add_argument("--dry-run", action="store_true", help="Print payload and exit without POSTing")

    args = p.parse_args(argv)

    api_key = resolve_api_key(args)
    registry_url = args.registry_url.rstrip("/")
    url = f"{registry_url}/models"

    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path).expanduser().resolve()
        info = parse_checkpoint_filename(ckpt_path.name)
        ckpt = CheckpointInfo(path=ckpt_path, step=info.step, is_final=info.is_final)
    else:
        ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
        ckpt = find_latest_checkpoint(ckpt_dir)

    payload = build_payload(args, ckpt=ckpt)

    if args.write_metadata:
        out_path = Path(args.write_metadata)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    if args.upload:
        artifact_uri = str(payload.get("artifact_uri") or "")
        if not artifact_uri.startswith("s3://"):
            raise SystemExit("--upload requires artifact_uri to be s3://...")
        endpoint_url = args.s3_endpoint_url or _env("ARTIFACT_S3_ENDPOINT_URL") or _env("AWS_S3_ENDPOINT_URL") or _env("AWS_ENDPOINT_URL")
        region = args.s3_region or _env("AWS_REGION") or "us-east-1"
        upload_to_s3(
            local_path=ckpt.path,
            artifact_uri=artifact_uri,
            size_bytes=int(payload["checkpoint_size_bytes"]),
            endpoint_url=endpoint_url,
            region=region,
            skip_if_exists=bool(args.skip_upload_if_exists),
            force=bool(args.force_upload),
        )

    result = post_json(url=url, api_key=api_key, payload=payload)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
