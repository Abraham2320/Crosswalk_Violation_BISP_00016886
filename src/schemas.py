from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


@dataclass(slots=True)
class ViolationTrigger:
    vehicle_id: int
    vehicle_zone: Optional[str]
    pedestrian_direction: str
    pedestrian_zone: Optional[str]
    reason: str


@dataclass(slots=True)
class ViolationEvent:
    violation_id: str
    timestamp: datetime
    vehicle_id: int
    frame_index: int
    vehicle_bbox: Tuple[int, int, int, int]
    vehicle_zone: Optional[str]
    polygon: List[Tuple[int, int]]
    pedestrian_direction: str
    pedestrian_zone: Optional[str]
    confidence: float
    location: str
    violation_type: str = "crosswalk_violation"
    severity: str = "HIGH"
    snapshot_path: Optional[str] = None
    vehicle_speed_estimate: Optional[float] = None

    @classmethod
    def create(
        cls,
        vehicle_id: int,
        frame_index: int,
        vehicle_bbox: Tuple[int, int, int, int],
        vehicle_zone: Optional[str],
        polygon: List[Tuple[int, int]],
        pedestrian_direction: str,
        pedestrian_zone: Optional[str],
        confidence: float,
        location: str,
        violation_type: str = "crosswalk_violation",
        severity: str = "HIGH",
        snapshot_path: Optional[str] = None,
        vehicle_speed_estimate: Optional[float] = None,
    ) -> "ViolationEvent":
        return cls(
            violation_id=str(uuid4()),
            timestamp=utc_now(),
            vehicle_id=vehicle_id,
            frame_index=frame_index,
            vehicle_bbox=vehicle_bbox,
            vehicle_zone=vehicle_zone,
            polygon=polygon,
            pedestrian_direction=pedestrian_direction,
            pedestrian_zone=pedestrian_zone,
            confidence=confidence,
            location=location,
            violation_type=violation_type,
            severity=severity,
            snapshot_path=snapshot_path,
            vehicle_speed_estimate=vehicle_speed_estimate,
        )

    def to_metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return _to_builtin(payload)


@dataclass(slots=True)
class EvidenceBundle:
    event: ViolationEvent
    frame_path: Path
    vehicle_crop_path: Path
    vehicle_crop_bbox: Tuple[int, int, int, int]
    frame_shape: Tuple[int, int, int]


@dataclass(slots=True)
class PlateDetectionResult:
    plate_bbox: Optional[Tuple[int, int, int, int]]
    plate_crop_path: Optional[Path]
    source: str
    confidence: float


@dataclass(slots=True)
class OCRResult:
    plate_text: Optional[str]
    confidence: float
    raw_text: str = ""
    accepted: bool = False


@dataclass(slots=True)
class ReportPayload:
    violation_id: str
    timestamp: str
    plate_number: str
    violation_type: str
    pedestrian_direction: str
    location: str
    location_code: str
    authority_name: str
    fine_amount: float


@dataclass(slots=True)
class ReportResult:
    report_json: Dict[str, Any]
    report_text: str
    invoice_path: Optional[Path] = None


@dataclass(slots=True)
class InvoiceRecordData:
    violation_id: str
    amount: float
    status: str
    pdf_path: str
