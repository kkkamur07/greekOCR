"""Contract JSON round-trip tests — no GPU or weight files required."""

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
    SegmentBlock,
    SegmentLine,
    SegmentRunRequest,
    SegmentRunResponse,
    TranscribeRunRequest,
    TranscribeRunResponse,
)
from ml.contracts.segment import SegmentGeometryKind
from pydantic import ValidationError


def _round_trip(model: object) -> dict:
    cls = type(model)
    payload = json.loads(cls.model_validate(model).model_dump_json())
    restored = cls.model_validate_json(json.dumps(payload))
    assert restored == model
    return payload


def test_segment_run_request_round_trip():
    request = SegmentRunRequest(
        registry_model_id="kraken-blla",
        image_bytes=b"\x89PNG\r\n",
        params={"refine": True},
    )
    payload = _round_trip(request)
    assert payload["image_bytes"] == base64.b64encode(b"\x89PNG\r\n").decode()


def test_segment_run_response_round_trip():
    response = SegmentRunResponse(
        blocks=[SegmentBlock(external_id="b1", order=0, box={"x": 0, "y": 0})],
        lines=[
            SegmentLine(
                external_id="l1",
                order=0,
                block_external_id="b1",
                baseline={"type": "LineString", "coordinates": [[0, 0], [10, 0]]},
                kind=SegmentGeometryKind.polygon,
                points=[[0, 0], [10, 0], [10, 5], [0, 5]],
                kraken_ceiling=[[0, 0], [12, 0], [12, 6], [0, 6]],
            )
        ],
    )
    _round_trip(response)


def test_transcribe_run_request_round_trip():
    request = TranscribeRunRequest(
        registry_model_id="greek-calamariv1",
        image_bytes=b"line-image-bytes",
    )
    _round_trip(request)


def test_transcribe_run_response_round_trip():
    text = "αβ"
    response = TranscribeRunResponse(
        text=text,
        confidence=0.91,
        character_confidences=[
            CharacterConfidence(char="α", confidence=0.9),
            CharacterConfidence(char="β", confidence=0.92),
        ],
    )
    _round_trip(response)


def test_transcribe_response_rejects_misaligned_character_confidences():
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


def test_job_callback_segment_done_round_trip():
    callback = JobCallbackRequest(
        ml_job_id=uuid4(),
        product_job_id=uuid4(),
        task=MLTask.segment,
        status=MLJobStatus.done,
        output=SegmentRunResponse(
            lines=[
                SegmentLine(
                    external_id="l1",
                    order=0,
                    baseline={"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                    points=[[1, 1], [2, 1], [2, 2], [1, 2]],
                )
            ]
        ),
    )
    _round_trip(callback)


def test_job_callback_transcribe_done_round_trip():
    text = "hi"
    callback = JobCallbackRequest(
        ml_job_id=uuid4(),
        product_job_id=uuid4(),
        task=MLTask.transcribe,
        status=MLJobStatus.done,
        output=TranscribeRunResponse(
            text=text,
            confidence=0.88,
            character_confidences=[
                CharacterConfidence(char="h", confidence=0.87),
                CharacterConfidence(char="i", confidence=0.89),
            ],
        ),
    )
    _round_trip(callback)


def test_job_callback_failed_round_trip():
    callback = JobCallbackRequest(
        ml_job_id=uuid4(),
        product_job_id=uuid4(),
        task=MLTask.transcribe,
        status=MLJobStatus.failed,
        error="weights not found in cache",
    )
    _round_trip(callback)


def test_job_callback_done_requires_output():
    with pytest.raises(ValidationError, match="require structured output"):
        JobCallbackRequest(
            ml_job_id=UUID("00000000-0000-0000-0000-000000000001"),
            product_job_id=UUID("00000000-0000-0000-0000-000000000002"),
            task=MLTask.segment,
            status=MLJobStatus.done,
        )


def test_job_callback_rejects_wrong_output_type():
    with pytest.raises(ValidationError, match="SegmentRunResponse"):
        JobCallbackRequest(
            ml_job_id=uuid4(),
            product_job_id=uuid4(),
            task=MLTask.segment,
            status=MLJobStatus.done,
            output=TranscribeRunResponse(
                text="x",
                confidence=0.5,
                character_confidences=[CharacterConfidence(char="x", confidence=0.5)],
            ),
        )
