"""Inference API DTOs."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from backend.inference.infrastructure.orm_models import InferenceTask, JobStatus, JobType


class InferenceModelResponse(BaseModel):
    id: UUID
    name: str
    provider: str
    task: InferenceTask
    artifact_ref: str
    default_params: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelBindingCreateRequest(BaseModel):
    task: InferenceTask
    model_id: UUID
    overrides: dict = Field(default_factory=dict)


class ModelBindingUpdateRequest(BaseModel):
    model_id: UUID | None = None
    overrides: dict | None = None


class ModelBindingResponse(BaseModel):
    id: UUID
    task: InferenceTask
    model_id: UUID
    project_id: UUID | None
    document_id: UUID | None
    document_part_id: UUID | None
    overrides: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class ResolvedModelBindingResponse(BaseModel):
    binding: ModelBindingResponse
    model: InferenceModelResponse
    effective_params: dict


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

    model_config = {"from_attributes": True}
