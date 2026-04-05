"""
generate_demo_data.py
---------------------
Seeds the SQLite database with 200 realistic sample crosswalk violation
records so the dashboard and chatbot can be demonstrated without a live
video session.

Usage:
    python generate_demo_data.py
"""
from __future__ import annotations

import random
import sqlite3
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

DB_PATH = Path(__file__).resolve().parent / "crosswalk_violations.db"

# ---------------------------------------------------------------------------
# Schema (mirrors src/storage/database.py – kept in sync manually)
# ---------------------------------------------------------------------------

DDL = """
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
    vehicle_image_path TEXT NOT NULL,
    frame_image_path TEXT NOT NULL,
    plate_image_path TEXT,
    report_path TEXT,
    invoice_path TEXT,
    violation_type TEXT NOT NULL,
    pedestrian_direction TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    location TEXT NOT NULL,
    created_at TEXT NOT NULL,
    llm_report_json TEXT,
    llm_report_text TEXT,
    vehicle_ref_id TEXT
);

CREATE TABLE IF NOT EXISTS invoices (
    id TEXT PRIMARY KEY,
    violation_id TEXT UNIQUE NOT NULL,
    amount REAL NOT NULL,
    issued_at TEXT NOT NULL,
    status TEXT NOT NULL,
    pdf_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_violations_plate_number
    ON violations (plate_number);
CREATE INDEX IF NOT EXISTS ix_violations_plate_timestamp
    ON violations (plate_number, timestamp);
"""

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

LOCATIONS = [
    "Crosswalk A – Amir Temur Ave",
    "Crosswalk B – Navoi St",
    "Crosswalk C – Mustaqillik Ave",
    "Crosswalk D – Shota Rustaveli St",
]

DIRECTIONS = ["UP", "DOWN", "STATIC"]

VEHICLE_IDS = list(range(1, 46))  # CAR_001 … CAR_045 (numeric IDs)

# Violation hour weights: peaks at 08:00 and 17:00–18:00 (rush hours)
HOUR_WEIGHTS = [
    1, 1, 1, 1, 1, 2,     # 00–05
    4, 7, 9, 6, 5, 5,     # 06–11
    6, 5, 5, 5, 6, 9,     # 12–17
    8, 6, 4, 3, 2, 1,     # 18–23
]


def _uzbek_plate() -> str:
    """Return a random Uzbekistan-format plate: 2 digits + letter + 3 digits + 2 letters.
    Example: 01A123BC
    """
    region = f"{random.randint(1, 99):02d}"
    mid_letter = random.choice(string.ascii_uppercase)
    digits = f"{random.randint(0, 999):03d}"
    suffix = "".join(random.choices(string.ascii_uppercase, k=2))
    return f"{region}{mid_letter}{digits}{suffix}"


def _random_timestamp(base: datetime, days_back: int = 7) -> datetime:
    """Random timestamp within the last `days_back` days, weighted by hour."""
    day_offset = timedelta(days=random.randint(0, days_back - 1))
    hour = random.choices(range(24), weights=HOUR_WEIGHTS)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    ts = base - day_offset
    return ts.replace(hour=hour, minute=minute, second=second, microsecond=0)


# ---------------------------------------------------------------------------
# Main seeding function
# ---------------------------------------------------------------------------

def _migrate(conn: sqlite3.Connection) -> None:
    """Add any columns that exist in the schema but are missing from an older DB."""
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(violations)").fetchall()
    }
    additions = {
        "created_at":         "TEXT",
        "llm_report_json":    "TEXT",
        "llm_report_text":    "TEXT",
        "vehicle_ref_id":     "TEXT",
        "plate_image_path":   "TEXT",
        "report_path":        "TEXT",
        "invoice_path":       "TEXT",
    }
    for col, col_type in additions.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE violations ADD COLUMN {col} {col_type}")
            print(f"  migrated: added column '{col}' to violations")
    conn.commit()


def seed(n: int = 200) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    conn.commit()
    _migrate(conn)

    now = datetime.now(timezone.utc)
    frame_counter = 1000

    # Pre-build a pool of plates (some vehicles have no plate)
    plates: dict[int, str | None] = {}
    for vid in VEHICLE_IDS:
        plates[vid] = _uzbek_plate() if random.random() < 0.6 else None

    inserted = 0
    for _ in range(n):
        vid = random.choice(VEHICLE_IDS)
        plate = plates[vid]
        ts = _random_timestamp(now)
        violation_id = str(uuid4())
        frame_counter += random.randint(5, 40)

        row = {
            "id": violation_id,
            "timestamp": ts.isoformat(),
            "plate_number": plate,
            "vehicle_id": vid,
            "vehicle_image_path": f"artifacts/vehicles/{violation_id}.jpg",
            "frame_image_path": f"artifacts/frames/{violation_id}.jpg",
            "plate_image_path": (
                f"artifacts/plates/{violation_id}.jpg" if plate else None
            ),
            "report_path": f"artifacts/reports/{violation_id}_report.json",
            "invoice_path": f"artifacts/invoices/{violation_id}.txt",
            "violation_type": "crosswalk_violation",
            "pedestrian_direction": random.choice(DIRECTIONS),
            "confidence": round(random.uniform(0.55, 0.95), 3),
            "status": "processed" if plate else "pending",
            "location": random.choice(LOCATIONS),
            "created_at": ts.isoformat(),
            "llm_report_json": None,
            "llm_report_text": None,
            "vehicle_ref_id": None,
        }

        conn.execute(
            """
            INSERT OR IGNORE INTO violations (
                id, timestamp, plate_number, vehicle_id,
                vehicle_image_path, frame_image_path, plate_image_path,
                report_path, invoice_path, violation_type, pedestrian_direction,
                confidence, status, location, created_at,
                llm_report_json, llm_report_text, vehicle_ref_id
            ) VALUES (
                :id, :timestamp, :plate_number, :vehicle_id,
                :vehicle_image_path, :frame_image_path, :plate_image_path,
                :report_path, :invoice_path, :violation_type, :pedestrian_direction,
                :confidence, :status, :location, :created_at,
                :llm_report_json, :llm_report_text, :vehicle_ref_id
            )
            """,
            row,
        )
        inserted += 1

        # Upsert vehicle record when plate is known
        if plate:
            # Detect whether the vehicles table has an 'id' column
            veh_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(vehicles)").fetchall()
            }
            existing = conn.execute(
                "SELECT violations_count FROM vehicles WHERE plate_number = ?",
                (plate,),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE vehicles SET violations_count = ? WHERE plate_number = ?",
                    (existing[0] + 1, plate),
                )
            else:
                if "id" in veh_cols:
                    conn.execute(
                        "INSERT INTO vehicles (id, plate_number, violations_count) VALUES (?, ?, 1)",
                        (str(uuid4()), plate),
                    )
                else:
                    conn.execute(
                        "INSERT INTO vehicles (plate_number, violations_count) VALUES (?, 1)",
                        (plate,),
                    )

    conn.commit()
    conn.close()

    total = conn_count()
    print(f"Done. Inserted {inserted} violations → database now has {total} records.")
    print(f"Database: {DB_PATH}")


def conn_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    conn.close()
    return n


if __name__ == "__main__":
    seed(200)
