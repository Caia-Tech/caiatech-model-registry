#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import publish_checkpoint as pc


def _read_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_stable_file(p: Path, *, stable_seconds: float) -> bool:
    try:
        st1 = p.stat()
    except FileNotFoundError:
        return False
    time.sleep(max(0.0, stable_seconds))
    try:
        st2 = p.stat()
    except FileNotFoundError:
        return False
    return (st1.st_size == st2.st_size) and (st1.st_mtime == st2.st_mtime)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Watch a checkpoint dir and auto-upload+register new checkpoints")
    ap.add_argument("--registry-url", default="http://localhost:8001")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--status", default="experimental", choices=["experimental", "staging", "production", "archived"])
    ap.add_argument("--artifact-uri-template", required=True)

    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--s3-endpoint-url", default=None)
    ap.add_argument("--s3-region", default=None)
    ap.add_argument("--skip-upload-if-exists", action="store_true")
    ap.add_argument("--force-upload", action="store_true")

    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--stable-seconds", type=float, default=2.0, help="Wait this long and re-stat before publishing")
    ap.add_argument("--state-file", default=None, help="JSON file to track last published step (default: <dir>/.publish_state.json)")
    ap.add_argument("--once", action="store_true", help="Publish at most one checkpoint and exit")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.exists():
        raise SystemExit(f"checkpoint-dir not found: {ckpt_dir}")

    state_path = Path(args.state_file).expanduser().resolve() if args.state_file else (ckpt_dir / ".publish_state.json")
    state = _read_state(state_path)
    last_step: Optional[int] = state.get("last_step") if isinstance(state.get("last_step"), int) else None

    while True:
        latest = pc.find_latest_checkpoint(ckpt_dir)
        step = latest.step
        if step is None:
            time.sleep(max(0.2, args.poll_seconds))
            if args.once:
                return 0
            continue

        if last_step is not None and step <= last_step:
            time.sleep(max(0.2, args.poll_seconds))
            if args.once:
                return 0
            continue

        if not _is_stable_file(latest.path, stable_seconds=args.stable_seconds):
            time.sleep(max(0.2, args.poll_seconds))
            continue

        cmd = [
            "--registry-url",
            args.registry_url,
            "--name",
            args.name,
            "--status",
            args.status,
            "--checkpoint-path",
            str(latest.path),
            "--artifact-uri-template",
            args.artifact_uri_template,
        ]
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.upload:
            cmd.append("--upload")
            if args.s3_endpoint_url:
                cmd.extend(["--s3-endpoint-url", args.s3_endpoint_url])
            if args.s3_region:
                cmd.extend(["--s3-region", args.s3_region])
            if args.skip_upload_if_exists:
                cmd.append("--skip-upload-if-exists")
            if args.force_upload:
                cmd.append("--force-upload")
        if args.dry_run:
            cmd.append("--dry-run")

        rc = pc.main(cmd)
        if rc == 0:
            last_step = step
            state["last_step"] = step
            state["last_path"] = str(latest.path)
            _write_state(state_path, state)

        if args.once:
            return rc

        time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))

