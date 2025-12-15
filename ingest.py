#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path


def sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


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


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Register a model artifact with Caia Model Registry")
    p.add_argument("--registry-url", default="http://localhost:8001", help="Base URL, e.g. http://localhost:8001")
    p.add_argument("--api-key", default=None, help="API key (or set CAIA_REGISTRY_API_KEY)")

    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--status", default="experimental", choices=["experimental", "staging", "production", "archived"])

    p.add_argument("--artifact-uri", default=None, help="Preferred. If omitted, defaults to sha256:<checkpoint_sha256>.")
    p.add_argument("--checkpoint-path", default=None, help="Local checkpoint file path (used to compute sha/size).")
    p.add_argument("--config-path", default=None, help="Local config JSON file path (used to compute sha/size + config_json).")
    p.add_argument(
        "--include-local-paths",
        action="store_true",
        help="Also send checkpoint_path/config_path to the server (dev-only servers must enable CAIA_REGISTRY_ALLOW_LOCAL_PATHS).",
    )

    p.add_argument("--run-id", default=None)
    p.add_argument("--git-commit", default=None)
    p.add_argument("--created-by", default=None)
    p.add_argument("--source-host", default=None)

    p.add_argument("--training-step", type=int, default=None)
    p.add_argument("--training-loss", type=float, default=None)
    p.add_argument("--dataset", default=None)

    p.add_argument("--description", default=None)
    p.add_argument("--tag", action="append", default=None, help="Repeatable, e.g. --tag onyx --tag hope")
    p.add_argument("--metrics-json", default=None, help="Raw JSON object string for metrics")
    p.add_argument("--metrics-path", default=None, help="Path to JSON file containing metrics object")

    p.add_argument("--write-metadata", default=None, help="Write the final JSON payload to this path")
    p.add_argument("--dry-run", action="store_true", help="Print payload and exit without POSTing")

    args = p.parse_args(argv)

    api_key = resolve_api_key(args)
    registry_url = args.registry_url.rstrip("/")
    url = f"{registry_url}/models"

    payload: dict = {
        "name": args.name,
        "version": args.version,
        "status": args.status,
        "run_id": args.run_id,
        "git_commit": args.git_commit,
        "created_by": args.created_by or os.getenv("USER") or None,
        "source_host": args.source_host or socket.gethostname(),
        "training_step": args.training_step,
        "training_loss": args.training_loss,
        "dataset": args.dataset,
        "description": args.description,
        "tags": args.tag,
    }

    if args.metrics_json and args.metrics_path:
        raise SystemExit("Use only one of --metrics-json or --metrics-path")
    if args.metrics_json:
        metrics = json.loads(args.metrics_json)
        if not isinstance(metrics, dict):
            raise SystemExit("--metrics-json must be a JSON object")
        payload["metrics"] = metrics
    elif args.metrics_path:
        payload["metrics"] = load_json_file(Path(args.metrics_path))

    checkpoint_sha256 = None
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        checkpoint_sha256, checkpoint_size = sha256_and_size(ckpt_path)
        payload["checkpoint_sha256"] = checkpoint_sha256
        payload["checkpoint_size_bytes"] = checkpoint_size
        if args.include_local_paths:
            payload["checkpoint_path"] = str(ckpt_path)

    if args.config_path:
        cfg_path = Path(args.config_path)
        cfg_sha256, cfg_size = sha256_and_size(cfg_path)
        payload["config_sha256"] = cfg_sha256
        payload["config_size_bytes"] = cfg_size
        payload["config_json"] = load_json_file(cfg_path)
        if args.include_local_paths:
            payload["config_path"] = str(cfg_path)

    artifact_uri = (args.artifact_uri or "").strip() or None
    if not artifact_uri:
        if checkpoint_sha256:
            artifact_uri = f"sha256:{checkpoint_sha256}"
        else:
            raise SystemExit("--artifact-uri is required unless --checkpoint-path is provided")
    payload["artifact_uri"] = artifact_uri

    if args.write_metadata:
        out_path = Path(args.write_metadata)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    result = post_json(url=url, api_key=api_key, payload=payload)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

