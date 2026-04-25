from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from uuid import uuid4

from config import AppSettings
from schemas import InvoiceRecordData

try:
    from sqlalchemy import DateTime, Float, ForeignKey, Integer, Numeric, String, Text, create_engine, func, select, text
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ModuleNotFoundError:
    SQLALCHEMY_AVAILABLE = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


if SQLALCHEMY_AVAILABLE:
    class Base(DeclarativeBase):
        pass


    class VehicleRecord(Base):
        __tablename__ = "vehicles"

        id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
        plate_number: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
        owner_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
        violations_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

        violations: Mapped[List["ViolationRecord"]] = relationship(back_populates="vehicle")


    class ViolationRecord(Base):
        __tablename__ = "violations"

        id: Mapped[str] = mapped_column(String(64), primary_key=True)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        plate_number: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
        vehicle_id: Mapped[int] = mapped_column(Integer, nullable=False)
        vehicle_image_path: Mapped[str] = mapped_column(Text, nullable=False)
        frame_image_path: Mapped[str] = mapped_column(Text, nullable=False)
        plate_image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        report_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        invoice_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        violation_type: Mapped[str] = mapped_column(String(64), nullable=False)
        severity: Mapped[str] = mapped_column(String(32), nullable=False, default="HIGH")
        pedestrian_direction: Mapped[str] = mapped_column(String(32), nullable=False)
        confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
        status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
        location: Mapped[str] = mapped_column(String(128), nullable=False)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
        llm_report_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        llm_report_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        vehicle_ref_id: Mapped[Optional[str]] = mapped_column(
            String(64),
            ForeignKey("vehicles.id", ondelete="SET NULL"),
            nullable=True,
        )
        vehicle_speed_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        snapshot_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        location_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
        plate_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

        vehicle: Mapped[Optional[VehicleRecord]] = relationship(back_populates="violations")
        invoice: Mapped[Optional["InvoiceRecord"]] = relationship(back_populates="violation", uselist=False)


    class InvoiceRecord(Base):
        __tablename__ = "invoices"

        id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
        violation_id: Mapped[str] = mapped_column(
            String(64),
            ForeignKey("violations.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        )
        amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
        issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
        status: Mapped[str] = mapped_column(String(32), default="issued", nullable=False)
        pdf_path: Mapped[str] = mapped_column(Text, nullable=False)

        violation: Mapped[ViolationRecord] = relationship(back_populates="invoice")


    class Database:
        def __init__(self, settings: AppSettings):
            self.settings = settings
            primary_url = self._resolve_database_url(settings)
            self.database_url, self.engine = self._build_engine_with_fallback(
                primary_url,
                settings.storage.sqlite_fallback_url,
            )
            self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

        def _build_engine(self, url: str):
            connect_args = {}
            if url.startswith("sqlite"):
                connect_args["check_same_thread"] = False
            elif url.startswith("postgresql"):
                # Add a short connection timeout to avoid hanging on unavailable PostgreSQL server
                connect_args["connect_timeout"] = 3
            return create_engine(url, future=True, connect_args=connect_args, pool_pre_ping=True, pool_recycle=3600)

        def _build_engine_with_fallback(self, primary_url: str, fallback_url: str):
            # Prefer configured DB, but gracefully fall back to SQLite when Postgres
            # is unavailable in local development (missing driver, no server, etc.).
            if primary_url.startswith("postgresql"):
                try:
                    engine = self._build_engine(primary_url)
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    return primary_url, engine
                except Exception as exc:
                    if not fallback_url.startswith("sqlite"):
                        fallback_url = "sqlite:///crosswalk_violations.db"
                    print(f"[Database] Primary DB unavailable; falling back to SQLite. ({exc})")
                    return fallback_url, self._build_engine(fallback_url)

            return primary_url, self._build_engine(primary_url)

        def _resolve_database_url(self, settings: AppSettings) -> str:
            url = settings.storage.database_url
            if url.startswith("postgresql") or url.startswith("sqlite"):
                return url
            return settings.storage.sqlite_fallback_url

        def create_all(self) -> None:
            Base.metadata.create_all(self.engine)

        @contextmanager
        def session(self) -> Iterator[Session]:
            session = self.SessionLocal()
            try:
                yield session
                session.commit()
            except SQLAlchemyError:
                session.rollback()
                raise
            finally:
                session.close()


    class ViolationRepository:
        def __init__(self, db: Database):
            self.db = db

        def upsert_vehicle(self, plate_number: str, owner_name: Optional[str] = None) -> VehicleRecord:
            with self.db.session() as session:
                vehicle = session.scalar(
                    select(VehicleRecord).where(VehicleRecord.plate_number == plate_number)
                )
                if vehicle is None:
                    vehicle = VehicleRecord(
                        plate_number=plate_number,
                        owner_name=owner_name,
                        violations_count=1,
                    )
                    session.add(vehicle)
                else:
                    vehicle.violations_count += 1
                    if owner_name and not vehicle.owner_name:
                        vehicle.owner_name = owner_name
                session.flush()
                return vehicle

        def save_violation(self, payload: Dict[str, object]) -> ViolationRecord:
            with self.db.session() as session:
                plate_number = payload.get("plate_number")
                vehicle = None
                if plate_number:
                    vehicle = session.scalar(
                        select(VehicleRecord).where(VehicleRecord.plate_number == plate_number)
                    )
                record = ViolationRecord(
                    **payload,
                    vehicle_ref_id=vehicle.id if vehicle is not None else None,
                )
                session.add(record)
                session.flush()
                return record

        def create_invoice(self, invoice: InvoiceRecordData) -> InvoiceRecord:
            with self.db.session() as session:
                record = InvoiceRecord(
                    violation_id=invoice.violation_id,
                    amount=Decimal(str(invoice.amount)),
                    status=invoice.status,
                    pdf_path=invoice.pdf_path,
                )
                session.add(record)
                session.flush()
                return record

        def list_violations(self, limit: int = 100) -> List[ViolationRecord]:
            with self.db.session() as session:
                return list(
                    session.scalars(
                        select(ViolationRecord)
                        .order_by(ViolationRecord.timestamp.desc())
                        .limit(limit)
                    )
                )

        def get_violation(self, violation_id: str) -> Optional[ViolationRecord]:
            with self.db.session() as session:
                return session.get(ViolationRecord, violation_id)

        def get_plate_history(self, plate_number: str) -> List[ViolationRecord]:
            with self.db.session() as session:
                return list(
                    session.scalars(
                        select(ViolationRecord)
                        .where(ViolationRecord.plate_number == plate_number)
                        .order_by(ViolationRecord.timestamp.desc())
                    )
                )

        def get_vehicle_by_plate(self, plate_number: str) -> Optional[VehicleRecord]:
            with self.db.session() as session:
                return session.scalar(
                    select(VehicleRecord).where(VehicleRecord.plate_number == plate_number)
                )

        def update_plate_number(
            self,
            violation_id: str,
            plate_number: Optional[str],
            confidence: float = 0.0,
        ) -> None:
            """Persist a deferred plate reading onto an already-saved violation record."""
            with self.db.session() as session:
                record = session.get(ViolationRecord, violation_id)
                if record is None:
                    return
                record.plate_number = plate_number
                if plate_number and plate_number not in ("UNREAD", "UNREADABLE"):
                    record.status = "processed"
                    vehicle = session.scalar(
                        select(VehicleRecord).where(VehicleRecord.plate_number == plate_number)
                    )
                    if vehicle is not None:
                        record.vehicle_ref_id = vehicle.id

        def analytics(self) -> Dict[str, object]:
            with self.db.session() as session:
                total = session.scalar(select(func.count(ViolationRecord.id))) or 0
                statuses = session.execute(
                    select(ViolationRecord.status, func.count(ViolationRecord.id)).group_by(
                        ViolationRecord.status
                    )
                ).all()
                per_day = session.execute(
                    select(func.date(ViolationRecord.timestamp), func.count(ViolationRecord.id))
                    .group_by(func.date(ViolationRecord.timestamp))
                    .order_by(func.date(ViolationRecord.timestamp))
                ).all()
                return {
                    "total_violations": total,
                    "by_status": [{"status": row[0], "count": row[1]} for row in statuses],
                    "by_day": [{"day": str(row[0]), "count": row[1]} for row in per_day],
                }

else:
    @dataclass(slots=True)
    class VehicleRecord:
        id: str
        plate_number: str
        owner_name: Optional[str]
        violations_count: int


    @dataclass(slots=True)
    class ViolationRecord:
        id: str
        timestamp: datetime
        plate_number: Optional[str]
        vehicle_id: int
        vehicle_image_path: str
        frame_image_path: str
        plate_image_path: Optional[str]
        report_path: Optional[str]
        invoice_path: Optional[str]
        violation_type: str
        severity: str
        pedestrian_direction: str
        confidence: float
        status: str
        location: str
        created_at: datetime
        llm_report_json: Optional[str]
        llm_report_text: Optional[str]
        vehicle_ref_id: Optional[str] = None
        vehicle_speed_estimate: Optional[float] = None
        snapshot_path: Optional[str] = None
        location_name: Optional[str] = None
        plate_crop_path: Optional[str] = None


    @dataclass(slots=True)
    class InvoiceRecord:
        id: str
        violation_id: str
        amount: Decimal
        issued_at: datetime
        status: str
        pdf_path: str


    class Database:
        def __init__(self, settings: AppSettings):
            self.db_path = self._resolve_sqlite_path(settings.storage.sqlite_fallback_url)

        def _resolve_sqlite_path(self, database_url: str) -> Path:
            if database_url.startswith("sqlite:///"):
                return Path(database_url.replace("sqlite:///", "", 1))
            return Path("crosswalk_violations.db")

        def create_all(self) -> None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vehicles (
                        id TEXT PRIMARY KEY,
                        plate_number TEXT UNIQUE NOT NULL,
                        owner_name TEXT,
                        violations_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
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
                        severity TEXT NOT NULL DEFAULT 'HIGH',
                        pedestrian_direction TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 0,
                        status TEXT NOT NULL,
                        location TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        llm_report_json TEXT,
                        llm_report_text TEXT,
                        vehicle_ref_id TEXT,
                        snapshot_path TEXT,
                        location_name TEXT,
                        vehicle_speed_estimate REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS invoices (
                        id TEXT PRIMARY KEY,
                        violation_id TEXT UNIQUE NOT NULL,
                        amount REAL NOT NULL,
                        issued_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        pdf_path TEXT NOT NULL
                    )
                    """
                )
                # Migration: add columns if DB was created before they existed
                for _col, _def in [
                    ("severity",               "TEXT NOT NULL DEFAULT 'HIGH'"),
                    ("snapshot_path",          "TEXT"),
                    ("location_name",          "TEXT"),
                    ("vehicle_speed_estimate", "REAL"),
                    ("plate_crop_path",        "TEXT"),
                ]:
                    try:
                        conn.execute(
                            f"ALTER TABLE violations ADD COLUMN {_col} {_def}"
                        )
                    except sqlite3.OperationalError:
                        pass  # column already exists
                conn.execute("CREATE INDEX IF NOT EXISTS ix_violations_plate_number ON violations (plate_number)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS ix_violations_plate_timestamp ON violations (plate_number, timestamp)"
                )
                conn.commit()

        @contextmanager
        def session(self) -> Iterator[sqlite3.Connection]:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


    class ViolationRepository:
        def __init__(self, db: Database):
            self.db = db

        def _row_to_vehicle(self, row: sqlite3.Row) -> VehicleRecord:
            return VehicleRecord(
                id=row["id"],
                plate_number=row["plate_number"],
                owner_name=row["owner_name"],
                violations_count=row["violations_count"],
            )

        def _row_to_violation(self, row: sqlite3.Row) -> ViolationRecord:
            return ViolationRecord(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                plate_number=row["plate_number"],
                vehicle_id=row["vehicle_id"],
                vehicle_image_path=row["vehicle_image_path"],
                frame_image_path=row["frame_image_path"],
                plate_image_path=row["plate_image_path"],
                report_path=row["report_path"],
                invoice_path=row["invoice_path"],
                violation_type=row["violation_type"],
                severity=row["severity"] if "severity" in row.keys() else "HIGH",
                pedestrian_direction=row["pedestrian_direction"],
                confidence=row["confidence"],
                status=row["status"],
                location=row["location"],
                created_at=datetime.fromisoformat(row["created_at"]),
                llm_report_json=row["llm_report_json"],
                llm_report_text=row["llm_report_text"],
                vehicle_ref_id=row["vehicle_ref_id"],
                snapshot_path=row["snapshot_path"] if "snapshot_path" in row.keys() else None,
                location_name=row["location_name"] if "location_name" in row.keys() else None,
                vehicle_speed_estimate=row["vehicle_speed_estimate"] if "vehicle_speed_estimate" in row.keys() else None,
                plate_crop_path=row["plate_crop_path"] if "plate_crop_path" in row.keys() else None,
            )

        def _row_to_invoice(self, row: sqlite3.Row) -> InvoiceRecord:
            return InvoiceRecord(
                id=row["id"],
                violation_id=row["violation_id"],
                amount=Decimal(str(row["amount"])),
                issued_at=datetime.fromisoformat(row["issued_at"]),
                status=row["status"],
                pdf_path=row["pdf_path"],
            )

        def upsert_vehicle(self, plate_number: str, owner_name: Optional[str] = None) -> VehicleRecord:
            with self.db.session() as conn:
                row = conn.execute(
                    "SELECT * FROM vehicles WHERE plate_number = ?",
                    (plate_number,),
                ).fetchone()
                if row is None:
                    vehicle = VehicleRecord(
                        id=str(uuid4()),
                        plate_number=plate_number,
                        owner_name=owner_name,
                        violations_count=1,
                    )
                    conn.execute(
                        """
                        INSERT INTO vehicles (id, plate_number, owner_name, violations_count)
                        VALUES (?, ?, ?, ?)
                        """,
                        (vehicle.id, vehicle.plate_number, vehicle.owner_name, vehicle.violations_count),
                    )
                    return vehicle

                updated = VehicleRecord(
                    id=row["id"],
                    plate_number=plate_number,
                    owner_name=row["owner_name"] or owner_name,
                    violations_count=int(row["violations_count"]) + 1,
                )
                conn.execute(
                    """
                    UPDATE vehicles
                    SET owner_name = ?, violations_count = ?
                    WHERE plate_number = ?
                    """,
                    (updated.owner_name, updated.violations_count, plate_number),
                )
                return updated

        def save_violation(self, payload: Dict[str, object]) -> ViolationRecord:
            timestamp = payload["timestamp"]
            created_at = payload.get("created_at") or utc_now()
            if isinstance(timestamp, datetime):
                timestamp_value = timestamp.isoformat()
            else:
                timestamp_value = str(timestamp)
            if isinstance(created_at, datetime):
                created_at_value = created_at.isoformat()
            else:
                created_at_value = str(created_at)

            vehicle_ref_id = None
            plate_number = payload.get("plate_number")
            if plate_number:
                vehicle = self.get_vehicle_by_plate(str(plate_number))
                vehicle_ref_id = vehicle.id if vehicle else None

            with self.db.session() as conn:
                conn.execute(
                    """
                    INSERT INTO violations (
                        id, timestamp, plate_number, vehicle_id, vehicle_image_path, frame_image_path,
                        plate_image_path, report_path, invoice_path, violation_type, severity,
                        pedestrian_direction, confidence, status, location, created_at,
                        llm_report_json, llm_report_text, vehicle_ref_id, snapshot_path,
                        vehicle_speed_estimate, plate_crop_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["id"],
                        timestamp_value,
                        payload.get("plate_number"),
                        payload["vehicle_id"],
                        payload["vehicle_image_path"],
                        payload["frame_image_path"],
                        payload.get("plate_image_path"),
                        payload.get("report_path"),
                        payload.get("invoice_path"),
                        payload["violation_type"],
                        payload.get("severity", "HIGH"),
                        payload["pedestrian_direction"],
                        payload["confidence"],
                        payload["status"],
                        payload["location"],
                        created_at_value,
                        payload.get("llm_report_json"),
                        payload.get("llm_report_text"),
                        vehicle_ref_id,
                        payload.get("snapshot_path"),
                        payload.get("vehicle_speed_estimate"),
                        payload.get("plate_crop_path"),
                    ),
                )
                row = conn.execute("SELECT * FROM violations WHERE id = ?", (payload["id"],)).fetchone()
                return self._row_to_violation(row)

        def create_invoice(self, invoice: InvoiceRecordData) -> InvoiceRecord:
            record = InvoiceRecord(
                id=str(uuid4()),
                violation_id=invoice.violation_id,
                amount=Decimal(str(invoice.amount)),
                issued_at=utc_now(),
                status=invoice.status,
                pdf_path=invoice.pdf_path,
            )
            with self.db.session() as conn:
                conn.execute(
                    """
                    INSERT INTO invoices (id, violation_id, amount, issued_at, status, pdf_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.violation_id,
                        float(record.amount),
                        record.issued_at.isoformat(),
                        record.status,
                        record.pdf_path,
                    ),
                )
            return record

        def list_violations(self, limit: int = 100) -> List[ViolationRecord]:
            with self.db.session() as conn:
                rows = conn.execute(
                    "SELECT * FROM violations ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [self._row_to_violation(row) for row in rows]

        def get_violation(self, violation_id: str) -> Optional[ViolationRecord]:
            with self.db.session() as conn:
                row = conn.execute(
                    "SELECT * FROM violations WHERE id = ?",
                    (violation_id,),
                ).fetchone()
                return self._row_to_violation(row) if row else None

        def get_plate_history(self, plate_number: str) -> List[ViolationRecord]:
            with self.db.session() as conn:
                rows = conn.execute(
                    "SELECT * FROM violations WHERE plate_number = ? ORDER BY timestamp DESC",
                    (plate_number,),
                ).fetchall()
                return [self._row_to_violation(row) for row in rows]

        def get_vehicle_by_plate(self, plate_number: str) -> Optional[VehicleRecord]:
            with self.db.session() as conn:
                row = conn.execute(
                    "SELECT * FROM vehicles WHERE plate_number = ?",
                    (plate_number,),
                ).fetchone()
                return self._row_to_vehicle(row) if row else None

        def update_plate_number(
            self,
            violation_id: str,
            plate_number: Optional[str],
            confidence: float = 0.0,
        ) -> None:
            """Persist a deferred plate reading onto an already-saved violation record."""
            is_real = bool(plate_number and plate_number not in ("UNREAD", "UNREADABLE"))
            status = "processed" if is_real else "pending"
            with self.db.session() as conn:
                vehicle_ref_id = None
                if is_real:
                    row = conn.execute(
                        "SELECT id FROM vehicles WHERE plate_number = ?", (plate_number,)
                    ).fetchone()
                    if row:
                        vehicle_ref_id = row["id"]
                # COALESCE keeps any existing vehicle_ref_id if we couldn't resolve one
                conn.execute(
                    """UPDATE violations
                       SET plate_number = ?,
                           status = ?,
                           vehicle_ref_id = COALESCE(?, vehicle_ref_id)
                       WHERE id = ?""",
                    (plate_number, status, vehicle_ref_id, violation_id),
                )

        def analytics(self) -> Dict[str, object]:
            with self.db.session() as conn:
                total = conn.execute("SELECT COUNT(*) AS count FROM violations").fetchone()["count"]
                statuses = conn.execute(
                    "SELECT status, COUNT(id) AS count FROM violations GROUP BY status"
                ).fetchall()
                per_day = conn.execute(
                    "SELECT substr(timestamp, 1, 10) AS day, COUNT(id) AS count FROM violations GROUP BY day ORDER BY day"
                ).fetchall()
                return {
                    "total_violations": total,
                    "by_status": [{"status": row["status"], "count": row["count"]} for row in statuses],
                    "by_day": [{"day": row["day"], "count": row["count"]} for row in per_day],
                }
