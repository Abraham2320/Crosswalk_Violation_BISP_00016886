from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query, status
from api.dependencies import get_repository
from api.models.schemas import ViolationCreate, ViolationResponse
from api.services.violation_service import ViolationService
from storage.database import ViolationRepository
router = APIRouter(prefix="/violations", tags=["violations"])
def get_service(repository: ViolationRepository = Depends(get_repository)) -> ViolationService:
    return ViolationService(repository)
def _g(record, attr: str, default=None):
    return getattr(record, attr, default)

def _serialize(record) -> ViolationResponse:
    return ViolationResponse(
        id=_g(record, "id", ""),
        timestamp=_g(record, "timestamp"),
        plate_number=_g(record, "plate_number"),
        vehicle_id=_g(record, "vehicle_id", 0),
        vehicle_image_path=_g(record, "vehicle_image_path"),
        frame_image_path=_g(record, "frame_image_path"),
        plate_image_path=_g(record, "plate_image_path"),
        report_path=_g(record, "report_path"),
        invoice_path=_g(record, "invoice_path"),
        violation_type=_g(record, "violation_type", "crosswalk_violation"),
        pedestrian_direction=_g(record, "pedestrian_direction"),
        confidence=_g(record, "confidence", 1.0),
        status=_g(record, "status", "pending"),
        location=_g(record, "location"),
        severity=_g(record, "severity"),
        snapshot_path=_g(record, "snapshot_path"),
        vehicle_speed_estimate=_g(record, "vehicle_speed_estimate"),
        created_at=_g(record, "created_at") or _g(record, "timestamp"),
        llm_report_text=_g(record, "llm_report_text"),
    )
@router.get("", response_model=list[ViolationResponse])
def list_violations(
    limit: int = Query(default=100, ge=1, le=500),
    service: ViolationService = Depends(get_service),
) -> list[ViolationResponse]:
    return [_serialize(record) for record in service.list_violations(limit=limit)]
@router.get("/{violation_id}", response_model=ViolationResponse)
def get_violation(
    violation_id: str,
    service: ViolationService = Depends(get_service),
) -> ViolationResponse:
    record = service.get_violation(violation_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Violation not found")
    return _serialize(record)
@router.post("", response_model=ViolationResponse, status_code=status.HTTP_201_CREATED)
def create_violation(
    payload: ViolationCreate,
    service: ViolationService = Depends(get_service),
) -> ViolationResponse:
    return _serialize(service.create_violation(payload))
