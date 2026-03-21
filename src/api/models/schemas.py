from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class InvoiceResponse(BaseModel):
    id: Optional[str] = None
    violation_id: str
    amount: float
    issued_at: Optional[datetime] = None
    status: str
    pdf_path: str


class ViolationCreate(BaseModel):
    id: str
    timestamp: datetime
    plate_number: Optional[str] = None
    vehicle_id: int
    vehicle_image_path: str
    frame_image_path: str
    plate_image_path: Optional[str] = None
    report_path: Optional[str] = None
    invoice_path: Optional[str] = None
    violation_type: str = "crosswalk_violation"
    pedestrian_direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = "pending"
    location: str
    llm_report_json: Optional[str] = None
    llm_report_text: Optional[str] = None


class ViolationResponse(BaseModel):
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
    pedestrian_direction: str
    confidence: float
    status: str
    location: str
    created_at: datetime
    llm_report_text: Optional[str]


class VehicleResponse(BaseModel):
    id: str
    plate_number: str
    owner_name: Optional[str]
    violations_count: int


class AnalyticsResponse(BaseModel):
    total_violations: int
    by_status: list[dict[str, object]]
    by_day: list[dict[str, object]]
