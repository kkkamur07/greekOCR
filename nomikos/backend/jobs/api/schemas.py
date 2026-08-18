"""Job API DTOs."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.domain.execution import ExecutionTarget


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
    # The **inference host** this job runs on, and the one the account setting
    # asked for. Both are on every representation of a job, which is what makes
    # "a failed job reports which host it failed on" true without a second
    # field: the host is stated whatever the status.
    execution_target: ExecutionTarget
    preferred_execution_target: ExecutionTarget
    # The preferred host had no **capacity**, so the job went to the other one.
    # Announced on the job, never in a transient toast — a researcher who looks
    # away must still be able to read where their work went. Issue 059 renders
    # it; this issue only has to make it true and readable.
    execution_target_substituted: bool = False
    # Retained for clients written against the pre-execution-target API. Now
    # derived from the column rather than the payload, so there is one source of
    # truth. ``local_only`` was never a value here and is not one now.
    execution: Literal["local", "cloud"] | None = None

    model_config = {"from_attributes": True}


def job_response_from_orm(job: Job) -> JobResponse:
    """Map ORM job to API DTO, naming the **inference host** that runs it."""
    response = JobResponse.model_validate(job)
    return response.model_copy(
        update={
            "execution_target_substituted": job.execution_target_substituted,
            "execution": job.execution_target.value,
        }
    )


class JobPageResponse(BaseModel):
    items: list[JobResponse]
    next_cursor: str | None = None


class ClearJobHistoryResponse(BaseModel):
    deleted: int
