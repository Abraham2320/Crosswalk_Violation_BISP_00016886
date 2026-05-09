from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_repository
from api.models.schemas import VehicleResponse
from api.services.violation_service import ViolationService
from storage.database import ViolationRepository
router = APIRouter(prefix="/vehicles", tags=["vehicles"])
def get_service(repository: ViolationRepository = Depends(get_repository)) -> ViolationService:
    return ViolationService(repository)
@router.get("/{plate_number}", response_model=VehicleResponse)
def get_vehicle(
    plate_number: str,
    service: ViolationService = Depends(get_service),
) -> VehicleResponse:
    vehicle = service.get_vehicle(plate_number)
    if vehicle is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vehicle not found")
    return VehicleResponse(
        id=vehicle.id,
        plate_number=vehicle.plate_number,
        owner_name=vehicle.owner_name,
        violations_count=vehicle.violations_count,
    )
