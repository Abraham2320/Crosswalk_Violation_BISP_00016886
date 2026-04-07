"""
generate_demo_data.py
─────────────────────
Seeds the SQLite database with 100 realistic crosswalk violation records
and generates demo snapshot images so the dashboard looks complete without
a live camera session.

Usage:
    python generate_demo_data.py
    python generate_demo_data.py --count 200   # custom record count
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

DB_PATH       = Path(__file__).resolve().parent / "crosswalk_violations.db"
SNAPSHOTS_DIR = Path(__file__).resolve().parent / "static" / "snapshots"

# ── Specific Uzbek plates (realistic format: DD + letter + DDD + 2 letters) ──
DEMO_PLATES = [
    # Region 01 (Tashkent city)
    "01A123BC", "01B456DE", "01C789FG", "01D321HJ", "01E654KL",
    "01F987MN", "01G159PQ", "01H260RS", "01J371TU", "01K482VW",
    # Region 30 (Tashkent oblast)
    "30A111HJ", "30B222KL", "30C333MN", "30D444PQ", "30E555RS",
    # Region 40 (Samarkand)
    "40A444PQ", "40B555RS", "40C666TU",
    # Region 50 (Namangan)
    "50A777VW", "50B888XY",
    # Region 60 (Andijan)
    "60A000BC", "60B111DE",
    # Region 70 (Fergana)
    "70A333HJ", "70B444KL",
    # Vehicles with no plate captured (None entries handled below)
]

# 20 % of vehicles have no plate captured
PLATE_POOL   = DEMO_PLATES + [None] * (len(DEMO_PLATES) // 4)

# ── Locations with GPS (Tashkent crosswalks) ─────────────────────────────────
LOCATIONS = [
    {
        "name":    "Crosswalk A – Amir Temur Ave",
        "lat":     41.2963,
        "lng":     69.2798,
        "address": "Amir Temur Avenue, near Tashkent City Mall",
    },
    {
        "name":    "Crosswalk B – Navoi St",
        "lat":     41.2959,
        "lng":     69.2697,
        "address": "Alisher Navoi Street, near Fine Arts Museum",
    },
    {
        "name":    "Crosswalk C – Mustaqillik Ave",
        "lat":     41.3003,
        "lng":     69.2726,
        "address": "Mustaqillik Avenue, near Independence Square",
    },
    {
        "name":    "Crosswalk D – Shota Rustaveli St",
        "lat":     41.2890,
        "lng":     69.2625,
        "address": "Shota Rustaveli Street, near State Conservatory",
    },
]

DIRECTIONS   = ["UP", "DOWN", "STATIC"]
VEHICLE_IDS  = list(range(1, 31))

HOUR_WEIGHTS = [
    1, 1, 1, 1, 1, 2,
    4, 7, 9, 6, 5, 5,
    6, 5, 5, 5, 6, 9,
    8, 6, 4, 3, 2, 1,
]

# ── Schema ────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS vehicles (
    id TEXT PRIMARY KEY,
    plate_number TEXT UNIQUE NOT NULL,
    owner_name TEXT,
    violations_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS violations (
    id                       TEXT PRIMARY KEY,
    timestamp                TEXT NOT NULL,
    plate_number             TEXT,
    vehicle_id               INTEGER NOT NULL,
    vehicle_image_path       TEXT NOT NULL,
    frame_image_path         TEXT NOT NULL,
    plate_image_path         TEXT,
    report_path              TEXT,
    invoice_path             TEXT,
    violation_type           TEXT NOT NULL,
    severity                 TEXT NOT NULL DEFAULT 'HIGH',
    pedestrian_direction     TEXT NOT NULL,
    confidence               REAL NOT NULL DEFAULT 0,
    status                   TEXT NOT NULL,
    location                 TEXT NOT NULL,
    created_at               TEXT NOT NULL,
    llm_report_json          TEXT,
    llm_report_text          TEXT,
    vehicle_ref_id           TEXT,
    snapshot_path            TEXT,
    location_name            TEXT,
    vehicle_speed_estimate   REAL,
    latitude                 REAL,
    longitude                REAL,
    location_address         TEXT
);

CREATE TABLE IF NOT EXISTS invoices (
    id TEXT PRIMARY KEY,
    violation_id TEXT UNIQUE NOT NULL,
    amount REAL NOT NULL,
    issued_at TEXT NOT NULL,
    status TEXT NOT NULL,
    pdf_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_violations_plate_number ON violations (plate_number);
CREATE INDEX IF NOT EXISTS ix_violations_plate_timestamp ON violations (plate_number, timestamp);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(violations)").fetchall()
    }
    additions = {
        "created_at":              "TEXT",
        "llm_report_json":         "TEXT",
        "llm_report_text":         "TEXT",
        "vehicle_ref_id":          "TEXT",
        "plate_image_path":        "TEXT",
        "report_path":             "TEXT",
        "invoice_path":            "TEXT",
        "severity":                "TEXT NOT NULL DEFAULT 'HIGH'",
        "snapshot_path":           "TEXT",
        "location_name":           "TEXT",
        "vehicle_speed_estimate":  "REAL",
        "latitude":                "REAL",
        "longitude":               "REAL",
        "location_address":        "TEXT",
    }
    for col, col_type in additions.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE violations ADD COLUMN {col} {col_type}")
            print(f"  migrated: added column '{col}'")
    conn.commit()


# ── Demo snapshot generator (uses PIL if available) ───────────────────────────

def _make_snapshot(violation_id: str, plate: str | None,
                   vtype: str, ts: str, location: str) -> str | None:
    """Generate a demo JPEG and return its relative path or None."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io

        W, H = 640, 360
        img  = Image.new("RGB", (W, H), (15, 17, 23))
        draw = ImageDraw.Draw(img)

        # Red semi-transparent banner
        banner = Image.new("RGBA", (W, 40), (180, 0, 0, 153))
        img.paste(banner, (0, 0))
        draw = ImageDraw.Draw(img)

        def txt(x, y, text, size=14, color=(255, 255, 255)):
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except Exception:
                font = ImageFont.load_default()
            draw.text((x, y), text, fill=color, font=font)

        txt(10, 10, "VIOLATION DETECTED", size=15, color=(255, 255, 255))

        # Crosswalk zone outline (amber)
        draw.polygon([(160, 80), (480, 80), (480, 280), (160, 280)],
                     outline=(11, 158, 245))

        # Plate-colored box (yellow, like real Uzbek plate)
        draw.rectangle([220, 130, 420, 180], fill=(255, 220, 0), outline=(0,0,0), width=2)
        txt(228, 138, plate or "UNDETECTED", size=22, color=(0, 0, 0))

        # Vehicle bounding box (red)
        draw.rectangle([200, 100, 440, 260], outline=(0, 0, 239), width=3)

        # Text overlay
        txt(10, 200, f"Type: {vtype}", size=13, color=(245, 158, 11))
        txt(10, 220, f"Time: {ts[:19].replace('T',' ')}", size=12, color=(180, 180, 180))
        txt(10, 240, f"Loc:  {location[:40]}", size=12, color=(180, 180, 180))
        txt(10, 335, "DEMO DATA — WIUT Crosswalk Enforcement System",
            size=11, color=(60, 60, 60))

        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"snapshot_{violation_id}_demo.jpg"
        path  = SNAPSHOTS_DIR / fname
        img.save(str(path), "JPEG", quality=85)
        return f"snapshots/{fname}"
    except ImportError:
        return None   # PIL not installed — skip silently
    except Exception as e:
        print(f"  [warn] snapshot generation failed: {e}")
        return None


def _random_ts(base: datetime, days_back: int = 30) -> datetime:
    day_offset = timedelta(days=random.randint(0, days_back - 1))
    hour  = random.choices(range(24), weights=HOUR_WEIGHTS)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return (base - day_offset).replace(
        hour=hour, minute=minute, second=second, microsecond=0
    )


# ── Main seeding function ─────────────────────────────────────────────────────

def seed(n: int = 100) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    conn.commit()
    _migrate(conn)

    now   = datetime.now(timezone.utc)
    inserted = 0

    for _ in range(n):
        vid   = random.choice(VEHICLE_IDS)
        plate = random.choice(PLATE_POOL)
        loc   = random.choice(LOCATIONS)
        ts    = _random_ts(now)
        vid_str = str(uuid4())

        if random.random() < 0.78:
            vtype, vsev = "FAILED_TO_YIELD", "HIGH"
        else:
            vtype, vsev = "UNSAFE_REENTRY", "LOW"

        snap = _make_snapshot(vid_str, plate, vtype, ts.isoformat(), loc["name"])
        speed = round(random.uniform(2.5, 18.0), 2)

        row = {
            "id":                     vid_str,
            "timestamp":              ts.isoformat(),
            "plate_number":           plate,
            "vehicle_id":             vid,
            "vehicle_image_path":     f"artifacts/vehicles/{vid_str}.jpg",
            "frame_image_path":       f"artifacts/frames/{vid_str}.jpg",
            "plate_image_path":       (f"artifacts/plates/{vid_str}.jpg" if plate else None),
            "report_path":            f"artifacts/reports/{vid_str}_report.json",
            "invoice_path":           f"artifacts/invoices/{vid_str}.txt",
            "violation_type":         vtype,
            "severity":               vsev,
            "pedestrian_direction":   random.choice(DIRECTIONS),
            "confidence":             round(random.uniform(0.60, 0.97), 3),
            "status":                 "processed" if plate else "pending",
            "location":               loc["name"],
            "created_at":             ts.isoformat(),
            "llm_report_json":        None,
            "llm_report_text":        None,
            "vehicle_ref_id":         None,
            "snapshot_path":          snap,
            "location_name":          loc["name"],
            "vehicle_speed_estimate": speed,
            "latitude":               loc["lat"] + random.uniform(-0.0003, 0.0003),
            "longitude":              loc["lng"] + random.uniform(-0.0003, 0.0003),
            "location_address":       loc["address"],
        }

        conn.execute(
            """
            INSERT OR IGNORE INTO violations (
                id, timestamp, plate_number, vehicle_id,
                vehicle_image_path, frame_image_path, plate_image_path,
                report_path, invoice_path, violation_type, severity,
                pedestrian_direction, confidence, status, location, created_at,
                llm_report_json, llm_report_text, vehicle_ref_id,
                snapshot_path, location_name, vehicle_speed_estimate,
                latitude, longitude, location_address
            ) VALUES (
                :id, :timestamp, :plate_number, :vehicle_id,
                :vehicle_image_path, :frame_image_path, :plate_image_path,
                :report_path, :invoice_path, :violation_type, :severity,
                :pedestrian_direction, :confidence, :status, :location, :created_at,
                :llm_report_json, :llm_report_text, :vehicle_ref_id,
                :snapshot_path, :location_name, :vehicle_speed_estimate,
                :latitude, :longitude, :location_address
            )
            """,
            row,
        )
        inserted += 1

        if plate:
            veh_cols = {r[1] for r in conn.execute("PRAGMA table_info(vehicles)").fetchall()}
            existing = conn.execute(
                "SELECT violations_count FROM vehicles WHERE plate_number = ?", (plate,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE vehicles SET violations_count = ? WHERE plate_number = ?",
                    (existing[0] + 1, plate),
                )
            elif "id" in veh_cols:
                conn.execute(
                    "INSERT INTO vehicles (id, plate_number, violations_count) VALUES (?,?,1)",
                    (str(uuid4()), plate),
                )
            else:
                conn.execute(
                    "INSERT INTO vehicles (plate_number, violations_count) VALUES (?,1)",
                    (plate,),
                )

    conn.commit()
    conn.close()

    total = _count()
    print(f"\nDone. Inserted {inserted} violations → database now has {total} records.")
    print(f"Snapshots: {SNAPSHOTS_DIR}")
    print(f"Database:  {DB_PATH}")
    print("\nSample plates you can search in the portal:")
    for p in DEMO_PLATES[:6]:
        print(f"  {p}")


def _count() -> int:
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    conn.close()
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    seed(args.count)
