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
from inference.contracts import (
    CharacterConfidence,
    JobCallbackRequest,
    JobSubmitRequest,
    JobSubmitResponse,
    InferenceJobStatus,
    InferenceTask,
    SegmentJobOutput,
    SegmentLine,
    SegmentRunResponse,
    TranscribeJobOutput,
    TranscribeBatchLineResult,
    TranscribeBatchRunResponse,
    TranscribeRunResponse,
)
from inference.contracts.run import InferenceRunRequest
from inference.contracts.segment import SegmentGeometryKind
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


def _transcribe_batch_response() -> TranscribeBatchRunResponse:
    return TranscribeBatchRunResponse(
        lines=[
            TranscribeBatchLineResult(
                line_id="line-1",
                line_index=0,
                output=_transcribe_response("α"),
            ),
            TranscribeBatchLineResult(
                line_id="line-2",
                line_index=1,
                output=_transcribe_response("β"),
            ),
        ]
    )


# --- image_bytes wire format ---
# Tests base64 encoding and validation. Does not run inference.


def test_image_bytes_json_contract():
    request = InferenceRunRequest(
        task=InferenceTask.segment,
        registry_model_id="kraken-blla",
        image_bytes=b"\x89PNG\r\n",
        params={"refine": True},
    )
    payload = _round_trip(request)
    assert payload["image_bytes"] == base64.b64encode(b"\x89PNG\r\n").decode()


def test_image_bytes_accepts_whitespace_wrapped_base64():
    encoded = base64.b64encode(b"\x89PNG\r\n").decode()
    request = InferenceRunRequest(
        task=InferenceTask.segment,
        registry_model_id="kraken-blla",
        image_bytes=f"\n{encoded[:4]} {encoded[4:]}\t",
    )

    assert request.image_bytes == b"\x89PNG\r\n"


def test_image_bytes_rejects_invalid_base64():
    with pytest.raises(ValidationError, match="valid base64"):
        InferenceRunRequest(
            task=InferenceTask.segment,
            registry_model_id="kraken-blla",
            image_bytes="not-base64!!!",
        )


# --- Segment and transcribe response contracts ---
# Tests JSON shape and field validation. Does not load models.


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


def test_transcribe_batch_response_json_contract():
    payload = _round_trip(_transcribe_batch_response())
    assert payload["lines"][0]["line_id"] == "line-1"
    assert payload["lines"][0]["output"]["text"] == "α"


# --- Job submit contracts ---
# Tests async job request/response serialization. Does not touch the database.


def test_job_submit_round_trip():
    product_job_id = uuid4()
    request = JobSubmitRequest(
        task=InferenceTask.segment,
        registry_model_id="kraken-blla",
        product_job_id=product_job_id,
        image_bytes=b"page-bytes",
    )
    payload = _round_trip(request)
    assert payload["product_job_id"] == str(product_job_id)

    response = JobSubmitResponse(inference_job_id=uuid4())
    _round_trip(response)


# --- Job submit validation rules ---
# Tests unsupported tasks and transcribe line requirements. Does not persist jobs.


def test_job_submit_rejects_tasks_without_contracts():
    with pytest.raises(ValidationError, match="unsupported job task: binarize"):
        JobSubmitRequest(
            task=InferenceTask.binarize,
            registry_model_id="future-binarizer",
            product_job_id=uuid4(),
            image_bytes=b"page-bytes",
        )


def test_async_transcribe_submit_requires_batched_line_regions():
    with pytest.raises(ValidationError, match="non-empty params.lines"):
        JobSubmitRequest(
            task=InferenceTask.transcribe,
            registry_model_id="syriac-calamariv1",
            product_job_id=uuid4(),
            image_bytes=b"page-bytes",
        )


# --- Job callback contracts ---
# Tests done/failed payloads and discriminated output unions. Does not POST to a server.


@pytest.mark.parametrize(
    ("task", "output_kind", "output"),
    [
        (
            InferenceTask.segment,
            "segment",
            SegmentJobOutput(kind="segment", data=_segment_response()),
        ),
        (
            InferenceTask.transcribe,
            "transcribe",
            TranscribeJobOutput(kind="transcribe", data=_transcribe_batch_response()),
        ),
    ],
)
def test_job_callback_done_output_union_round_trip(task, output_kind, output):
    callback = JobCallbackRequest(
        inference_job_id=uuid4(),
        product_job_id=uuid4(),
        task=task,
        status=InferenceJobStatus.done,
        output=output,
    )
    payload = _round_trip(callback)
    assert payload["output"]["kind"] == output_kind


def test_job_callback_failed_round_trip():
    callback = JobCallbackRequest(
        inference_job_id=uuid4(),
        product_job_id=uuid4(),
        task=InferenceTask.transcribe,
        status=InferenceJobStatus.failed,
        error="weights not found in cache",
    )
    _round_trip(callback)


def test_job_callback_rejects_invalid_terminal_payloads():
    with pytest.raises(ValidationError, match="unsupported job task: binarize"):
        JobCallbackRequest(
            inference_job_id=uuid4(),
            product_job_id=uuid4(),
            task=InferenceTask.binarize,
            status=InferenceJobStatus.failed,
            error="unsupported",
        )

    with pytest.raises(ValidationError, match="require structured output"):
        JobCallbackRequest(
            inference_job_id=UUID("00000000-0000-0000-0000-000000000001"),
            product_job_id=UUID("00000000-0000-0000-0000-000000000002"),
            task=InferenceTask.segment,
            status=InferenceJobStatus.done,
        )

    with pytest.raises(ValidationError, match="segment task requires segment output"):
        JobCallbackRequest(
            inference_job_id=uuid4(),
            product_job_id=uuid4(),
            task=InferenceTask.segment,
            status=InferenceJobStatus.done,
            output=TranscribeJobOutput(
                kind="transcribe",
                data=_transcribe_batch_response(),
            ),
        )

    with pytest.raises(ValidationError):
        JobCallbackRequest(
            inference_job_id=uuid4(),
            product_job_id=uuid4(),
            task=InferenceTask.transcribe,
            status=InferenceJobStatus.done,
            output=TranscribeJobOutput(
                kind="transcribe",
                data=_transcribe_response("x"),
            ),
        )
