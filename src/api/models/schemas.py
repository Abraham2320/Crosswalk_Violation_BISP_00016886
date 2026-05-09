from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class InvoiceResponse(BaseModel):
    id: Optional[str] = None
    violation_id: str
    amount: float
    issued_at: Optional[datetime] = None
    status: str
    pdf_path: Optional[str] = None


class ViolationCreate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    plate_number: Optional[str] = None
    vehicle_id: int
    vehicle_image_path: str = ""
    frame_image_path: str = ""
    plate_image_path: Optional[str] = None
    report_path: Optional[str] = None
    invoice_path: Optional[str] = None
    violation_type: str = "crosswalk_violation"
    pedestrian_direction: str = "UNKNOWN"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str = "pending"
    location: str = ""
    severity: str = "HIGH"
    snapshot_path: Optional[str] = None
    vehicle_speed_estimate: Optional[float] = None
    llm_report_json: Optional[str] = None
    llm_report_text: Optional[str] = None


class ViolationResponse(BaseModel):
    id: str
    timestamp: datetime
    plate_number: Optional[str] = None
    vehicle_id: int
    vehicle_image_path: Optional[str] = None
    frame_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    report_path: Optional[str] = None
    invoice_path: Optional[str] = None
    violation_type: str
    pedestrian_direction: Optional[str] = None
    confidence: float
    status: str
    location: Optional[str] = None
    severity: Optional[str] = None
    snapshot_path: Optional[str] = None
    vehicle_speed_estimate: Optional[float] = None
    created_at: Optional[datetime] = None
    llm_report_text: Optional[str] = None


class VehicleResponse(BaseModel):
    id: str
    plate_number: str
    owner_name: Optional[str] = None
    violations_count: int


class AnalyticsResponse(BaseModel):
    total_violations: int
    by_status: list[dict[str, object]]
    by_day: list[dict[str, object]]
