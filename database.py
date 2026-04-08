"""
database.py — Flask-specific database access layer.
Wraps the existing SQLite crosswalk_violations.db, adds migration,
admin_users, and audit_log tables, and seeds the default admin account.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from werkzeug.security import generate_password_hash

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "crosswalk_violations.db"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema migration + seed
# ---------------------------------------------------------------------------

def _existing_columns(conn: sqlite3.Connection, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def init_db() -> None:
    """Idempotent migration: adds new columns and tables, seeds default admin."""
    with db_connection() as conn:
        # ── violations: add new columns safely ───────────────────────────────
        cols = _existing_columns(conn, "violations")
        for col, col_type in {
            "snapshot_path":          "TEXT",
            "location_name":          "TEXT",
            "vehicle_speed_estimate": "REAL",
            "latitude":               "REAL",
            "longitude":              "REAL",
            "location_address":       "TEXT",
            "plate_crop_path":        "TEXT",
        }.items():
            if col not in cols:
                conn.execute(
                    f"ALTER TABLE violations ADD COLUMN {col} {col_type}"
                )

        # ── admin_users table ────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── audit_log table ──────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_username TEXT,
                action         TEXT,
                target         TEXT,
                timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── default admin (once) ─────────────────────────────────────────────
        row = conn.execute(
            "SELECT id FROM admin_users WHERE username = ?", ("admin",)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin1234")),
            )


# ---------------------------------------------------------------------------
# Audit logging helper
# ---------------------------------------------------------------------------

def log_audit(action: str, target: str = "", username: str = "system") -> None:
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO audit_log (admin_username, action, target) VALUES (?, ?, ?)",
            (username, action, str(target)),
        )
