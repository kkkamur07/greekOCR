"""Async job submit and completion callback contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from ml.contracts.common import ImageBytes, MLJobStatus, MLTask
from ml.contracts.segment import SegmentRunResponse
from ml.contracts.transcribe import TranscribeRunResponse

SUPPORTED_JOB_TASKS = frozenset({MLTask.segment, MLTask.transcribe})


class JobSubmitRequest(BaseModel):
    task: MLTask
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    product_job_id: UUID
    image_bytes: ImageBytes
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_supported_task(self) -> JobSubmitRequest:
        if self.task not in SUPPORTED_JOB_TASKS:
            raise ValueError(f"unsupported job task: {self.task.value}")
        return self


class JobSubmitResponse(BaseModel):
    ml_job_id: UUID


class SegmentJobOutput(BaseModel):
    kind: Literal["segment"]
    data: SegmentRunResponse


class TranscribeJobOutput(BaseModel):
    kind: Literal["transcribe"]
    data: TranscribeRunResponse


JobOutput = Annotated[
    SegmentJobOutput | TranscribeJobOutput,
    Field(discriminator="kind"),
]


class JobCallbackRequest(BaseModel):
    ml_job_id: UUID
    product_job_id: UUID
    task: MLTask
    status: MLJobStatus
    output: JobOutput | None = None
    error: str | None = None

    # These are request-body contract errors. FastAPI returns them as 422
    # validation responses when this model is used as an endpoint body.
    @model_validator(mode="after")
    def validate_terminal_payload(self) -> JobCallbackRequest:
        if self.task not in SUPPORTED_JOB_TASKS:
            raise ValueError(f"unsupported job task: {self.task.value}")

        if self.status == MLJobStatus.done:
            if self.output is None:
                raise ValueError("done callbacks require structured output")
            if self.error is not None:
                raise ValueError("done callbacks must not include error")
            if self.output.kind != self.task.value:
                raise ValueError(f"{self.task.value} task requires {self.task.value} output")
            return self

        if self.status == MLJobStatus.failed:
            if not self.error:
                raise ValueError("failed callbacks require error message")
            if self.output is not None:
                raise ValueError("failed callbacks must not include output")
            return self
        raise ValueError("callback status must be done or failed")

__all__ = [
    "JobCallbackRequest",
    "JobOutput",
    "JobSubmitRequest",
    "JobSubmitResponse",
    "SUPPORTED_JOB_TASKS",
    "SegmentJobOutput",
    "TranscribeJobOutput",
]
