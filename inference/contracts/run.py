"""Unified sync run request/response for POST /inference/v1/run."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from inference.admission import validate_request_params
from inference.contracts.common import ImageBytes, InferenceTask
from inference.contracts.segment import SegmentRunResponse
from inference.contracts.transcribe import TranscribeBatchRunResponse, TranscribeRunResponse
from inference.infrastructure.settings import get_inference_settings


class InferenceRunRequest(BaseModel):
    task: InferenceTask
    registry_model_id: str = Field(min_length=1)
    registry_tag: str = Field(default="stable", min_length=1)
    image_bytes: ImageBytes
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_admission_limits(self) -> InferenceRunRequest:
        validate_request_params(self.params, get_inference_settings())
        return self


class InferenceRunResponse(BaseModel):
    task: InferenceTask
    output: SegmentRunResponse | TranscribeRunResponse | TranscribeBatchRunResponse

    @model_validator(mode="after")
    def validate_output_matches_task(self) -> InferenceRunResponse:
        if self.task == InferenceTask.segment and not isinstance(self.output, SegmentRunResponse):
            raise ValueError("segment task requires SegmentRunResponse output")
        if self.task == InferenceTask.transcribe and not isinstance(
            self.output, (TranscribeRunResponse, TranscribeBatchRunResponse)
        ):
            raise ValueError("transcribe task requires TranscribeRunResponse output")
        return self
