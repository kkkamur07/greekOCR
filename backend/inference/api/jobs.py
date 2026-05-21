"""Job routes — test enqueue and status polling."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.inference.api.schemas import EnqueueTestJobRequest, EnqueueTestJobResponse, JobResponse
from backend.inference.application.job_service import JobService
from infrastructure.db import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_service(db: AsyncSession = Depends(get_db)) -> JobService:
    return JobService(db)


@router.post("/test", response_model=EnqueueTestJobResponse, status_code=201)
async def enqueue_test_job(
    body: EnqueueTestJobRequest | None = None,
    service: JobService = Depends(_job_service),
) -> EnqueueTestJobResponse:
    handler = body.handler if body else "noop"
    job = await service.enqueue_test_job(handler)
    return EnqueueTestJobResponse(job_id=job.id)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    service: JobService = Depends(_job_service),
) -> JobResponse:
    job = await service.get_job(job_id)
    return JobResponse.model_validate(job)
