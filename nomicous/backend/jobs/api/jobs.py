"""Job routes — test enqueue and status polling."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.settings.job import get_job_settings
from backend.jobs.api.schemas import EnqueueTestJobRequest, EnqueueTestJobResponse, JobResponse
from backend.jobs.application.job_service import JobService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_service(db: AsyncSession = Depends(get_db)) -> JobService:
    return JobService(db)


def _require_test_routes_enabled() -> None:
    if not get_job_settings().enable_test_job_routes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


@router.post("/test", response_model=EnqueueTestJobResponse, status_code=201)
async def enqueue_test_job(
    body: EnqueueTestJobRequest | None = None,
    service: JobService = Depends(_job_service),
    _enabled: None = Depends(_require_test_routes_enabled),
    current_user: User = Depends(get_current_user),
) -> EnqueueTestJobResponse:
    handler = body.handler if body else "noop"
    job = await service.enqueue_test_job(handler, user_id=current_user.id)
    return EnqueueTestJobResponse(job_id=job.id)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    service: JobService = Depends(_job_service),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    job = await service.get_job(job_id)
    if job.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return JobResponse.model_validate(job)
