#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str = ""
    severity: str = "FAIL"  # FAIL|WARN|INFO


def _is_git_repo(root: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return r.returncode == 0 and r.stdout.strip() == "true"


def _git_ls_files(root: Path) -> list[str]:
    r = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    raw = r.stdout.decode("utf-8", errors="replace")
    return [p for p in raw.split("\x00") if p]


def _walk_text_files(root: Path) -> Iterable[Path]:
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "qa_artifacts",
    }
    exts = {".py", ".md", ".yml", ".yaml", ".txt", ".sh", ".example"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            p = Path(dirpath) / filename
            if p.suffix.lower() in exts:
                yield p


def _scan_for_secrets(files: Iterable[Path]) -> list[str]:
    patterns = [
        ("private_key", re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----")),
        ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
        ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b")),
        ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ]
    hits: list[str] = []
    for path in files:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:4096]:
            continue
        text = data.decode("utf-8", errors="replace")
        for name, pat in patterns:
            if pat.search(text):
                hits.append(f"{name}: {path.as_posix()}")
    return hits


def main(argv: list[str]) -> int:  # noqa: ARG001
    root = Path(__file__).resolve().parent.parent
    results: list[CheckResult] = []

    git_repo = _is_git_repo(root)
    tracked_files: set[str] = set()
    if git_repo:
        tracked_files = set(_git_ls_files(root))
        results.append(CheckResult("git_repo", True, "OK", severity="INFO"))
    else:
        results.append(CheckResult("git_repo", True, "Not a git repo (tracked-file checks skipped)", severity="WARN"))

    def tracked(path: str) -> bool:
        return path in tracked_files

    env_path = root / ".env"
    if env_path.exists():
        if git_repo and tracked(".env"):
            results.append(CheckResult(".env tracked", False, "Remove .env from git history", severity="FAIL"))
        else:
            results.append(CheckResult(".env present", True, "Present on disk (ensure it is not committed)", severity="WARN"))
    else:
        results.append(CheckResult(".env present", True, "OK", severity="INFO"))

    registry_db = root / "registry.db"
    if registry_db.exists():
        if git_repo and tracked("registry.db"):
            results.append(CheckResult("registry.db tracked", False, "Remove registry.db from git history", severity="FAIL"))
        else:
            results.append(CheckResult("registry.db present", True, "Present on disk (should not be committed)", severity="WARN"))
    else:
        results.append(CheckResult("registry.db present", True, "OK", severity="INFO"))

    if git_repo:
        qa_tracked = any(p.startswith("qa_artifacts/") or p == "qa_artifacts" for p in tracked_files)
        if qa_tracked:
            results.append(
                CheckResult("qa_artifacts tracked", False, "Remove qa_artifacts/* from git history", severity="FAIL")
            )
        else:
            results.append(CheckResult("qa_artifacts tracked", True, "OK", severity="INFO"))
    else:
        results.append(CheckResult("qa_artifacts tracked", True, "Not a git repo (cannot verify tracked files)", severity="WARN"))

    secret_hits = _scan_for_secrets(_walk_text_files(root))
    if secret_hits:
        results.append(CheckResult("secret scan", False, "; ".join(secret_hits[:5]), severity="FAIL"))
    else:
        results.append(CheckResult("secret scan", True, "OK", severity="INFO"))

    failed = any((not r.ok) and r.severity == "FAIL" for r in results)
    for r in results:
        status = "PASS" if r.ok and r.severity != "WARN" else ("WARN" if r.severity == "WARN" else "FAIL")
        detail = f" - {r.details}" if r.details else ""
        sys.stdout.write(f"[{status}] {r.name}{detail}\n")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
