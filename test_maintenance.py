from __future__ import annotations

import sqlite3
from pathlib import Path


def test_maintenance_backup_and_vacuum(tmp_path: Path):
    import maintenance

    db_path = tmp_path / "src.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t (v) VALUES ('x')")
        conn.commit()
    finally:
        conn.close()

    backup_path = tmp_path / "backup.db"
    assert maintenance.main(["--db-path", str(db_path), "--backup", str(backup_path), "--vacuum"]) == 0

    conn = sqlite3.connect(backup_path)
    try:
        row = conn.execute("SELECT v FROM t").fetchone()
        assert row[0] == "x"
    finally:
        conn.close()

