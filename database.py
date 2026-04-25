"""
database.py — Flask-specific database access layer.
Wraps the existing SQLite crosswalk_violations.db, adds migration,
admin_users, and audit_log tables, and seeds the default admin account.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from werkzeug.security import generate_password_hash

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_db_path() -> Path:
    explicit_path = os.getenv("DB_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path)

    for env_name in ("DATABASE_URL", "SQLITE_FALLBACK_URL"):
        raw_url = os.getenv(env_name, "").strip()
        if raw_url.startswith("sqlite:///"):
            return Path(raw_url.replace("sqlite:///", "", 1))

    return PROJECT_ROOT / "crosswalk_violations.db"


DB_PATH = _resolve_db_path()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id TEXT PRIMARY KEY,
                plate_number TEXT UNIQUE NOT NULL,
                owner_name TEXT,
                violations_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                plate_number TEXT,
                vehicle_id INTEGER NOT NULL,
                vehicle_image_path TEXT NOT NULL DEFAULT '',
                frame_image_path TEXT NOT NULL DEFAULT '',
                plate_image_path TEXT,
                report_path TEXT,
                invoice_path TEXT,
                violation_type TEXT NOT NULL DEFAULT 'UNKNOWN',
                severity TEXT NOT NULL DEFAULT 'HIGH',
                pedestrian_direction TEXT NOT NULL DEFAULT 'STATIC',
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                location TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                llm_report_json TEXT,
                llm_report_text TEXT,
                vehicle_ref_id TEXT,
                snapshot_path TEXT,
                location_name TEXT,
                vehicle_speed_estimate REAL,
                latitude REAL,
                longitude REAL,
                location_address TEXT
            );

            CREATE TABLE IF NOT EXISTS invoices (
                id TEXT PRIMARY KEY,
                violation_id TEXT UNIQUE NOT NULL,
                amount REAL NOT NULL,
                issued_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'issued',
                pdf_path TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS ix_violations_plate_number ON violations (plate_number);
            CREATE INDEX IF NOT EXISTS ix_violations_plate_timestamp ON violations (plate_number, timestamp);
            """
        )

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

        # ── cameras table (admin-manageable camera registry) ─────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                cam_id         TEXT UNIQUE NOT NULL,
                label          TEXT NOT NULL,
                default_source TEXT NOT NULL DEFAULT '0',
                demo_source    TEXT NOT NULL DEFAULT '',
                location_name  TEXT NOT NULL DEFAULT 'Crosswalk A',
                latitude       REAL NOT NULL DEFAULT 41.2963,
                longitude      REAL NOT NULL DEFAULT 69.2798,
                tags           TEXT NOT NULL DEFAULT '',
                is_active      INTEGER NOT NULL DEFAULT 1,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure camera metadata columns exist for older DBs.
        cam_cols = _existing_columns(conn, "cameras")
        for col, col_type in {
            "location_name": "TEXT NOT NULL DEFAULT 'Crosswalk A'",
            "latitude": "REAL NOT NULL DEFAULT 41.2963",
            "longitude": "REAL NOT NULL DEFAULT 69.2798",
            "tags": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if col not in cam_cols:
                conn.execute(f"ALTER TABLE cameras ADD COLUMN {col} {col_type}")

        # ── default admin (once) ─────────────────────────────────────────────
        row = conn.execute(
            "SELECT id FROM admin_users WHERE username = ?", ("admin",)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin1234")),
            )

        # ── default cameras (once) ───────────────────────────────────────────
        cam_count = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        if cam_count == 0:
            conn.executemany(
                """
                INSERT INTO cameras (
                    cam_id, label, default_source, demo_source,
                    location_name, latitude, longitude, tags, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                [
                    ("cam1", "Camera 1 - Entrance", "0", "Videos/v1.mp4", "Entrance Gate", 41.2963, 69.2798, "entrance, north"),
                    ("cam2", "Camera 2 - Crosswalk", "0", "Videos/v2.mp4", "Main Crosswalk", 41.2965, 69.2801, "crosswalk, school-zone"),
                    ("cam3", "Camera 3 - Exit", "0", "Videos/v3.mp4", "Exit Lane", 41.2961, 69.2794, "exit, south"),
                ],
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
