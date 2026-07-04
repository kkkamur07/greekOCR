"""Unified sync run request/response for POST /ml/v1/run."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from ml_service.contracts.common import ImageBytes, MLTask
from ml_service.contracts.segment import SegmentRunResponse
from ml_service.contracts.transcribe import TranscribeRunResponse


class MlRunRequest(BaseModel):
    task: MLTask
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    image_bytes: ImageBytes
    params: dict[str, Any] = Field(default_factory=dict)


class MlRunResponse(BaseModel):
    task: MLTask
    output: SegmentRunResponse | TranscribeRunResponse

    @model_validator(mode="after")
    def validate_output_matches_task(self) -> MlRunResponse:
        if self.task == MLTask.segment and not isinstance(self.output, SegmentRunResponse):
            raise ValueError("segment task requires SegmentRunResponse output")
        if self.task == MLTask.transcribe and not isinstance(self.output, TranscribeRunResponse):
            raise ValueError("transcribe task requires TranscribeRunResponse output")
        return self
