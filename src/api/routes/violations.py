from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_repository
from api.models.schemas import ViolationCreate, ViolationResponse
from api.services.violation_service import ViolationService
from storage.database import ViolationRepository


router = APIRouter(prefix="/violations", tags=["violations"])


def get_service(repository: ViolationRepository = Depends(get_repository)) -> ViolationService:
    return ViolationService(repository)


def _serialize(record) -> ViolationResponse:
    return ViolationResponse(
        id=record.id,
        timestamp=record.timestamp,
        plate_number=record.plate_number,
        vehicle_id=record.vehicle_id,
        vehicle_image_path=record.vehicle_image_path,
        frame_image_path=record.frame_image_path,
        plate_image_path=record.plate_image_path,
        report_path=record.report_path,
        invoice_path=record.invoice_path,
        violation_type=record.violation_type,
        pedestrian_direction=record.pedestrian_direction,
        confidence=record.confidence,
        status=record.status,
        location=record.location,
        created_at=record.created_at,
        llm_report_text=record.llm_report_text,
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
