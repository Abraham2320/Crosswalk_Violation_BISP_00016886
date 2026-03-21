from __future__ import annotations

from typing import Optional

from api.models.schemas import ViolationCreate
from storage.database import ViolationRecord, ViolationRepository


class ViolationService:
    def __init__(self, repository: ViolationRepository):
        self.repository = repository

    def list_violations(self, limit: int = 100) -> list[ViolationRecord]:
        return self.repository.list_violations(limit=limit)

    def get_violation(self, violation_id: str) -> Optional[ViolationRecord]:
        return self.repository.get_violation(violation_id)

    def create_violation(self, payload: ViolationCreate) -> ViolationRecord:
        return self.repository.save_violation(
            {
                "id": payload.id,
                "timestamp": payload.timestamp,
                "plate_number": payload.plate_number,
                "vehicle_id": payload.vehicle_id,
                "vehicle_image_path": payload.vehicle_image_path,
                "frame_image_path": payload.frame_image_path,
                "plate_image_path": payload.plate_image_path,
                "report_path": payload.report_path,
                "invoice_path": payload.invoice_path,
                "violation_type": payload.violation_type,
                "pedestrian_direction": payload.pedestrian_direction,
                "confidence": payload.confidence,
                "status": payload.status,
                "location": payload.location,
                "llm_report_json": payload.llm_report_json,
                "llm_report_text": payload.llm_report_text,
            }
        )

    def get_vehicle(self, plate_number: str):
        return self.repository.get_vehicle_by_plate(plate_number)

    def analytics(self) -> dict[str, object]:
        return self.repository.analytics()
