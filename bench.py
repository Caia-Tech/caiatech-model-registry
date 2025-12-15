#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional


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


def http_json(method: str, url: str, *, api_key: str, body: Optional[dict] = None) -> Any:
    data = None
    headers = {"X-API-Key": api_key}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {detail}") from e


def load_model(registry_url: str, api_key: str, *, model_id: Optional[int], name: Optional[str], version: Optional[str]) -> dict:
    base = registry_url.rstrip("/")
    if model_id is not None:
        return http_json("GET", f"{base}/models/{model_id}", api_key=api_key)
    if not name or not version:
        raise SystemExit("Pass --model-id or both --name and --version")
    name_enc = urllib.parse.quote(name, safe="")
    version_enc = urllib.parse.quote(version, safe="")
    return http_json("GET", f"{base}/models/by-name/{name_enc}/{version_enc}", api_key=api_key)


def compute_smoke_metrics(model: dict, *, checkpoint_path: Optional[str], config_path: Optional[str]) -> dict:
    checkpoint_path = checkpoint_path or model.get("local_checkpoint_path")
    config_path = config_path or model.get("local_config_path")

    artifact_accessible = False
    checkpoint_size_bytes = model.get("checkpoint_size_bytes")
    if checkpoint_path:
        try:
            checkpoint_size_bytes = Path(checkpoint_path).stat().st_size
            artifact_accessible = True
        except OSError:
            artifact_accessible = False

    config_parse_ok = False
    if config_path:
        try:
            with Path(config_path).open("r", encoding="utf-8") as f:
                json.load(f)
            config_parse_ok = True
        except Exception:
            config_parse_ok = False

    score = 1.0 if config_parse_ok else 0.0
    return {
        "artifact_accessible": bool(artifact_accessible),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "config_parse_ok": bool(config_parse_ok),
        "score": score,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Run a lightweight benchmark and write metrics to Caia Model Registry")
    p.add_argument("--registry-url", default="http://localhost:8001")
    p.add_argument("--api-key", default=None)

    p.add_argument("--name", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--model-id", type=int, default=None)

    p.add_argument("--suite", required=True, choices=["smoke-v1", "core-v1"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--eval-commit", default=None)
    p.add_argument("--notes", default=None)

    p.add_argument("--checkpoint-path", default=None, help="Optional local checkpoint path for this runner")
    p.add_argument("--config-path", default=None, help="Optional local config path for this runner")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args(argv)
    api_key = resolve_api_key(args)

    if args.model_id is None and (not args.name or not args.version):
        raise SystemExit("Pass --model-id or both --name and --version")
    if args.model_id is not None and (args.name or args.version):
        raise SystemExit("Use either --model-id or --name/--version (not both)")

    model = load_model(
        args.registry_url,
        api_key,
        model_id=args.model_id,
        name=args.name,
        version=args.version,
    )

    suite = args.suite
    if suite in {"smoke-v1", "core-v1"}:
        metrics = compute_smoke_metrics(model, checkpoint_path=args.checkpoint_path, config_path=args.config_path)
    else:
        raise SystemExit(f"Unsupported suite: {suite}")

    payload: dict = {
        "suite": suite,
        "metrics": metrics,
        "dataset": args.dataset,
        "eval_commit": args.eval_commit,
        "notes": args.notes,
        "eval_config": {
            "runner": "bench.py",
            "suite": suite,
            "uses_local_paths": {"checkpoint": bool(args.checkpoint_path), "config": bool(args.config_path)},
        },
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    if args.dry_run:
        sys.stdout.write(json.dumps({"model_id": model["id"], "payload": payload}, indent=2) + "\n")
        return 0

    base = args.registry_url.rstrip("/")
    out = http_json("POST", f"{base}/models/{model['id']}/metrics", api_key=api_key, body=payload)
    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
