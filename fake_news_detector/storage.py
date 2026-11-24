"""SQLite history of past checks."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "history.db"
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            snippet TEXT NOT NULL,
            verdict TEXT NOT NULL,
            factuality REAL,
            result_json TEXT NOT NULL
        )"""
    )
    return conn


def save(result: dict) -> int:
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO checks (created_at, snippet, verdict, factuality, result_json) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                result["text"][:160],
                result["verdict"],
                result["factuality"],
                json.dumps(result),
            ),
        )
        return cur.lastrowid


def recent(limit: int = 20) -> list[dict]:
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, created_at, snippet, verdict, factuality FROM checks ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get(check_id: int) -> dict | None:
    with _lock, _conn() as conn:
        row = conn.execute("SELECT result_json FROM checks WHERE id = ?", (check_id,)).fetchone()
    return json.loads(row["result_json"]) if row else None
