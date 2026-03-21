from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_repository
from api.models.schemas import AnalyticsResponse
from api.services.violation_service import ViolationService
from storage.database import ViolationRepository


router = APIRouter(tags=["analytics"])


def get_service(repository: ViolationRepository = Depends(get_repository)) -> ViolationService:
    return ViolationService(repository)


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(service: ViolationService = Depends(get_service)) -> AnalyticsResponse:
    return AnalyticsResponse(**service.analytics())
