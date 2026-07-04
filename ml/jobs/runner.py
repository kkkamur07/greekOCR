"""v1 mock inference runner -- structured echo output until real runners land."""

from __future__ import annotations

from ml.architectures.mock import mock_segment
from ml.contracts.common import MLTask
from ml.contracts.segment import SegmentRunResponse
from ml.contracts.transcribe import CharacterConfidence, TranscribeRunResponse

from ml.infrastructure.orm_models import MLJob

#! This is not a production behaviour, remove when you have actual code to test.
def run_mock(job: MLJob) -> SegmentRunResponse | TranscribeRunResponse:
    if job.task == MLTask.segment:
        return mock_segment(job.image_bytes)
    if job.task == MLTask.transcribe:
        text = f"mock:{len(job.image_bytes)}"
        return TranscribeRunResponse(
            text=text,
            confidence=1.0,
            character_confidences=[
                CharacterConfidence(char=char, confidence=1.0) for char in text
            ],
        )
    raise ValueError(f"unsupported ML task for mock runner: {job.task}")


def run_job(job: MLJob) -> SegmentRunResponse | TranscribeRunResponse:
    return run_mock(job)
