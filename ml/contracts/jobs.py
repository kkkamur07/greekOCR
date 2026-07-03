"""Async job submit and completion callback contracts."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from ml.contracts.common import ImageBytes, MLJobStatus, MLTask
from ml.contracts.segment import SegmentRunResponse
from ml.contracts.transcribe import TranscribeRunResponse


class JobSubmitRequest(BaseModel):
    task: MLTask
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    product_job_id: UUID
    image_bytes: ImageBytes
    params: dict[str, Any] = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    ml_job_id: UUID


class JobCallbackRequest(BaseModel):
    ml_job_id: UUID
    product_job_id: UUID
    task: MLTask
    status: MLJobStatus
    output: SegmentRunResponse | TranscribeRunResponse | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> JobCallbackRequest:
        if self.status == MLJobStatus.done:
            if self.output is None:
                raise ValueError("done callbacks require structured output")
            if self.error is not None:
                raise ValueError("done callbacks must not include error")
            self._validate_output_matches_task()
            return self
        if self.status == MLJobStatus.failed:
            if not self.error:
                raise ValueError("failed callbacks require error message")
            if self.output is not None:
                raise ValueError("failed callbacks must not include output")
            return self
        raise ValueError("callback status must be done or failed")

    def _validate_output_matches_task(self) -> None:
        if self.output is None:
            return
        if self.task == MLTask.segment and not isinstance(self.output, SegmentRunResponse):
            raise ValueError("segment task requires SegmentRunResponse output")
        if self.task == MLTask.transcribe and not isinstance(self.output, TranscribeRunResponse):
            raise ValueError("transcribe task requires TranscribeRunResponse output")
