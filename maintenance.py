#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def default_db_path() -> Path:
    raw = os.getenv("CAIA_REGISTRY_DB_PATH")
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / "registry.db"


def connect(db_path: Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=busy_timeout_ms / 1000, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
    return conn


def backup_db(src_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    src = connect(src_path)
    try:
        dest = connect(dest_path)
        try:
            src.backup(dest)
            dest.commit()
        finally:
            dest.close()
    finally:
        src.close()


def vacuum_db(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        conn.execute("VACUUM;")
        conn.commit()
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="SQLite maintenance for Caia Model Registry")
    p.add_argument("--db-path", default=None, help="Path to SQLite DB (default: env CAIA_REGISTRY_DB_PATH or ./registry.db)")
    p.add_argument("--backup", default=None, help="Write a SQLite backup to this path")
    p.add_argument("--vacuum", action="store_true", help="Run VACUUM on the DB (may take time and locks the DB)")
    args = p.parse_args(argv)

    db_path = Path(args.db_path) if args.db_path else default_db_path()

    if not args.backup and not args.vacuum:
        raise SystemExit("Nothing to do: pass --backup <path> and/or --vacuum")

    if args.backup:
        backup_db(db_path, Path(args.backup))
        sys.stdout.write(f"Backup written: {args.backup}\n")

    if args.vacuum:
        vacuum_db(db_path)
        sys.stdout.write("VACUUM complete\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
