"""Job API DTOs."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType


class EnqueueTestJobRequest(BaseModel):
    handler: str = Field(default="noop", pattern="^(noop|fail)$")


class EnqueueTestJobResponse(BaseModel):
    job_id: UUID


class EnqueueJobResponse(BaseModel):
    job_id: UUID


class JobResponse(BaseModel):
    id: UUID
    type: JobType
    status: JobStatus
    payload: dict
    result: dict | None
    error: str | None
    user_id: UUID | None
    document_id: UUID | None
    document_part_id: UUID | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    execution: Literal["local", "cloud"] | None = None

    model_config = {"from_attributes": True}


def job_response_from_orm(job: Job) -> JobResponse:
    """Map ORM job to API DTO, including execution host from payload."""
    response = JobResponse.model_validate(job)
    payload = job.payload or {}
    execution = payload.get("execution")
    if execution not in ("local", "cloud"):
        execution = "cloud"
    return response.model_copy(update={"execution": execution})


class JobPageResponse(BaseModel):
    items: list[JobResponse]
    next_cursor: str | None = None
