"""Public ML contract tests.

These tests protect the JSON wire format shared by the ML API, ML worker,
and future Annote HTTP client. Keep this file focused on serialization,
discriminated unions, and custom validation rules; do not add tests that need
GPU, model weights, registry files, or runtime inference.
"""

from __future__ import annotations

import base64
import json
from uuid import UUID, uuid4

import pytest
from ml.contracts import (
    CharacterConfidence,
    JobCallbackRequest,
    JobSubmitRequest,
    JobSubmitResponse,
    MLJobStatus,
    MLTask,
    SegmentJobOutput,
    SegmentLine,
    SegmentRunRequest,
    SegmentRunResponse,
    TranscribeJobOutput,
    TranscribeRunResponse,
)
from ml.contracts.segment import SegmentGeometryKind
from pydantic import ValidationError

# Checks whether the payload survives the round trip. 
def _round_trip(model: object) -> dict:
    cls = type(model)
    payload = json.loads(cls.model_validate(model).model_dump_json())
    restored = cls.model_validate_json(json.dumps(payload))

    assert restored == model
    return payload


def _segment_response() -> SegmentRunResponse:
    return SegmentRunResponse(
        lines=[
            SegmentLine(
                external_id="l1",
                order=0,
                baseline={"type": "LineString", "coordinates": [[0, 0], [10, 0]]},
                kind=SegmentGeometryKind.polygon,
                points=[[0, 0], [10, 0], [10, 5], [0, 5]],
            )
        ],
    )


def _transcribe_response(text: str = "hi") -> TranscribeRunResponse:
    return TranscribeRunResponse(
        text=text,
        confidence=0.88,
        character_confidences=[
            CharacterConfidence(char=char, confidence=0.87)
            for char in text
        ],
    )


def test_image_bytes_json_contract():
    request = SegmentRunRequest(
        registry_model_id="kraken-blla",
        image_bytes=b"\x89PNG\r\n",
        params={"refine": True},
    )
    payload = _round_trip(request)
    assert payload["image_bytes"] == base64.b64encode(b"\x89PNG\r\n").decode()


def test_image_bytes_accepts_whitespace_wrapped_base64():
    encoded = base64.b64encode(b"\x89PNG\r\n").decode()
    request = SegmentRunRequest(
        registry_model_id="kraken-blla",
        image_bytes=f"\n{encoded[:4]} {encoded[4:]}\t",
    )

    assert request.image_bytes == b"\x89PNG\r\n"


def test_image_bytes_rejects_invalid_base64():
    with pytest.raises(ValidationError, match="valid base64"):
        SegmentRunRequest(registry_model_id="kraken-blla", image_bytes="not-base64!!!")


def test_segment_response_json_contract():
    payload = _round_trip(_segment_response())
    assert payload["lines"][0]["kind"] == "polygon"
    assert payload["lines"][0]["points"] == [[0, 0], [10, 0], [10, 5], [0, 5]]


def test_transcribe_response_json_contract_and_alignment_validation():
    payload = _round_trip(_transcribe_response("αβ"))
    assert payload["text"] == "αβ"

    with pytest.raises(ValidationError, match="length must match"):
        TranscribeRunResponse(
            text="ab",
            confidence=0.5,
            character_confidences=[CharacterConfidence(char="a", confidence=0.5)],
        )


def test_job_submit_round_trip():
    product_job_id = uuid4()
    request = JobSubmitRequest(
        task=MLTask.segment,
        registry_model_id="kraken-blla",
        product_job_id=product_job_id,
        image_bytes=b"page-bytes",
    )
    payload = _round_trip(request)
    assert payload["product_job_id"] == str(product_job_id)

    response = JobSubmitResponse(ml_job_id=uuid4())
    _round_trip(response)


def test_job_submit_rejects_tasks_without_contracts():
    with pytest.raises(ValidationError, match="unsupported job task: binarize"):
        JobSubmitRequest(
            task=MLTask.binarize,
            registry_model_id="future-binarizer",
            product_job_id=uuid4(),
            image_bytes=b"page-bytes",
        )


@pytest.mark.parametrize(
    ("task", "output_kind", "output"),
    [
        (
            MLTask.segment,
            "segment",
            SegmentJobOutput(kind="segment", data=_segment_response()),
        ),
        (
            MLTask.transcribe,
            "transcribe",
            TranscribeJobOutput(kind="transcribe", data=_transcribe_response()),
        ),
    ],
)
def test_job_callback_done_output_union_round_trip(task, output_kind, output):
    callback = JobCallbackRequest(
        ml_job_id=uuid4(),
        product_job_id=uuid4(),
        task=task,
        status=MLJobStatus.done,
        output=output,
    )
    payload = _round_trip(callback)
    assert payload["output"]["kind"] == output_kind


def test_job_callback_failed_round_trip():
    callback = JobCallbackRequest(
        ml_job_id=uuid4(),
        product_job_id=uuid4(),
        task=MLTask.transcribe,
        status=MLJobStatus.failed,
        error="weights not found in cache",
    )
    _round_trip(callback)


def test_job_callback_rejects_invalid_terminal_payloads():
    with pytest.raises(ValidationError, match="unsupported job task: binarize"):
        JobCallbackRequest(
            ml_job_id=uuid4(),
            product_job_id=uuid4(),
            task=MLTask.binarize,
            status=MLJobStatus.failed,
            error="unsupported",
        )

    with pytest.raises(ValidationError, match="require structured output"):
        JobCallbackRequest(
            ml_job_id=UUID("00000000-0000-0000-0000-000000000001"),
            product_job_id=UUID("00000000-0000-0000-0000-000000000002"),
            task=MLTask.segment,
            status=MLJobStatus.done,
        )

    with pytest.raises(ValidationError, match="segment task requires segment output"):
        JobCallbackRequest(
            ml_job_id=uuid4(),
            product_job_id=uuid4(),
            task=MLTask.segment,
            status=MLJobStatus.done,
            output=TranscribeJobOutput(
                kind="transcribe",
                data=_transcribe_response("x"),
            ),
        )
