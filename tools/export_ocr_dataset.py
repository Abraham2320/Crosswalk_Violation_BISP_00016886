from __future__ import annotations

import argparse
import random
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ALNUM = re.compile(r"[^A-Z0-9]")


@dataclass
class Row:
    violation_id: str
    plate_number: str
    plate_image_path: str
    vehicle_image_path: str


def clean_label(text: str) -> str:
    return ALNUM.sub("", (text or "").upper())


def load_rows(db_path: Path) -> list[Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
                        SELECT id, plate_number, plate_image_path, vehicle_image_path
            FROM violations
            WHERE plate_number IS NOT NULL
              AND TRIM(plate_number) != ''
            """
        ).fetchall()
    finally:
        conn.close()

    result: list[Row] = []
    for r in rows:
        result.append(
            Row(
                violation_id=str(r["id"]),
                plate_number=str(r["plate_number"]),
                plate_image_path=str(r["plate_image_path"] or ""),
                vehicle_image_path=str(r["vehicle_image_path"] or ""),
            )
        )
    return result


def resolve_path(project_root: Path, stored_path: str, violation_id: str, preferred: str) -> Path | None:
    """Resolve DB image paths even if they came from a different machine/container."""
    candidates: list[Path] = []
    raw = (stored_path or "").strip()
    if raw:
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((project_root / p).resolve())

        token = "/Crosswalk_Violation/"
        pos = raw.find(token)
        if pos >= 0:
            tail = raw[pos + len(token):].replace("\\", "/")
            candidates.append((project_root / tail).resolve())

        candidates.append((project_root / "artifacts" / "plates" / Path(raw).name).resolve())
        candidates.append((project_root / "artifacts" / "vehicles" / Path(raw).name).resolve())

    ext_candidates = [".jpg", ".jpeg", ".png", ".webp"]
    if preferred == "plate":
        for ext in ext_candidates:
            candidates.append((project_root / "artifacts" / "plates" / f"{violation_id}{ext}").resolve())
        for ext in ext_candidates:
            candidates.append((project_root / "artifacts" / "vehicles" / f"{violation_id}{ext}").resolve())
    else:
        for ext in ext_candidates:
            candidates.append((project_root / "artifacts" / "vehicles" / f"{violation_id}{ext}").resolve())
        for ext in ext_candidates:
            candidates.append((project_root / "artifacts" / "plates" / f"{violation_id}{ext}").resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_rows(rows: list[Row], val_ratio: float, seed: int) -> tuple[list[Row], list[Row]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows and val_rows:
        train_rows, val_rows = val_rows, []
    return train_rows, val_rows


def build_dataset(
    project_root: Path,
    rows: Iterable[Row],
    images_dir: Path,
    label_file: Path,
    min_label_len: int,
    max_label_len: int,
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    lines: list[str] = []

    for idx, row in enumerate(rows, start=1):
        label = clean_label(row.plate_number)
        if not label or not (min_label_len <= len(label) <= max_label_len):
            skipped += 1
            continue

        src = resolve_path(project_root, row.plate_image_path, row.violation_id, preferred="plate")
        if src is None:
            src = resolve_path(project_root, row.vehicle_image_path, row.violation_id, preferred="vehicle")
        if src is None or not src.exists() or not src.is_file():
            skipped += 1
            continue

        ext = src.suffix.lower() or ".jpg"
        dst_name = f"{idx:06d}_{row.violation_id}{ext}"
        dst = images_dir / dst_name
        shutil.copy2(src, dst)

        # PaddleOCR text format: relative_image_path<TAB>label
        lines.append(f"images/{dst_name}\t{label}")
        copied += 1

    label_file.write_text("\n".join(lines), encoding="utf-8")
    return copied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OCR training dataset from violations DB (plate crops + labels)."
    )
    parser.add_argument(
        "--db",
        default="crosswalk_violations.db",
        help="Path to SQLite database (default: crosswalk_violations.db)",
    )
    parser.add_argument(
        "--out",
        default="datasets/ocr_plate",
        help="Output dataset directory (default: datasets/ocr_plate)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (default: 42)",
    )
    parser.add_argument(
        "--min-label-len",
        type=int,
        default=5,
        help="Minimum plate label length after cleaning (default: 5)",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=10,
        help="Maximum plate label length after cleaning (default: 10)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    db_path = (project_root / args.db).resolve()
    out_root = (project_root / args.out).resolve()

    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return 1

    rows = load_rows(db_path)
    if not rows:
        print("[ERROR] No labeled plate rows found in violations table.")
        return 1

    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)

    images_dir = out_root / "images"
    ensure_dir(images_dir)

    train_txt = out_root / "train.txt"
    val_txt = out_root / "val.txt"

    train_copied, train_skipped = build_dataset(
        project_root, train_rows, images_dir, train_txt, args.min_label_len, args.max_label_len
    )
    val_copied, val_skipped = build_dataset(
        project_root, val_rows, images_dir, val_txt, args.min_label_len, args.max_label_len
    )

    dict_chars = sorted({ch for r in rows for ch in clean_label(r.plate_number)})
    dict_path = out_root / "dict.txt"
    dict_path.write_text("\n".join(dict_chars), encoding="utf-8")

    print("[OK] OCR dataset exported")
    print(f"  root: {out_root}")
    print(f"  train samples: {train_copied} (skipped: {train_skipped})")
    print(f"  val samples:   {val_copied} (skipped: {val_skipped})")
    print(f"  dict chars:    {len(dict_chars)} -> {dict_path}")
    print("  labels:        train.txt, val.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
