"""Job API DTOs."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from backend.inference.infrastructure.orm_models import JobStatus, JobType


class EnqueueTestJobRequest(BaseModel):
    handler: str = Field(default="noop", pattern="^(noop|fail)$")


class EnqueueTestJobResponse(BaseModel):
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

    model_config = {"from_attributes": True}
