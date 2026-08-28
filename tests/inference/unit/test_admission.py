"""Admission-control tests that do not load inference weights.

ADR 0002 deleted the loopback service, and with it the middleware that bounded a
request body and the secret that authenticated its callers. Everything asserted
here is what survived that deletion because it was never about the transport:
the limits sit next to the runtime, in ``inference.admission``, and are reached
from both sides of a **claim** - ``JobSubmitRequest`` on the platform's
submission path, and ``run_model`` in the **inference agent** that runs the page.

The HTTP status a violation used to produce is deliberately not asserted any
more. Nothing maps these failures to a status: an agent that cannot run a page
reports it failed, in words, through the existing job callback.
"""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image
from pydantic import ValidationError

from inference.admission import (
    CLIENT_INPUT_ERROR,
    validate_image_bytes,
    validate_request_params,
)
from inference.contracts.common import InferenceTask
from inference.contracts.jobs import JobSubmitRequest
from inference.settings import InferenceSettings, get_inference_settings


@pytest.fixture(autouse=True)
def clear_inference_settings_cache() -> None:
    get_inference_settings.cache_clear()
    yield
    get_inference_settings.cache_clear()


def _png_bytes(*, size: tuple[int, int] = (2, 2)) -> bytes:
    output = BytesIO()
    Image.new("L", size).save(output, format="PNG")
    return output.getvalue()


def _segment_job(**overrides) -> JobSubmitRequest:
    payload = {
        "task": InferenceTask.segment,
        "registry_model_id": "blla-segment",
        "product_job_id": uuid4(),
        "image_bytes": _png_bytes(),
        "params": {},
    }
    payload.update(overrides)
    return JobSubmitRequest(**payload)


# --- Image bounds ------------------------------------------------------------
# ``run_model`` calls ``validate_image_bytes`` before it resolves a model, so a
# page that would blow up the process is refused before any weights load.


def test_rejects_oversized_encoded_image_before_base64_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_MAX_ENCODED_IMAGE_BYTES", "1024")

    with pytest.raises(ValidationError, match=CLIENT_INPUT_ERROR):
        _segment_job(image_bytes="A" * 2_000)


def test_rejects_oversized_decoded_image_before_image_loading() -> None:
    settings = InferenceSettings(INFERENCE_MAX_DECODED_IMAGE_BYTES=1024, _env_file=None)

    with pytest.raises(ValueError, match=CLIENT_INPUT_ERROR):
        validate_image_bytes(b"x" * 1025, settings)


def test_rejects_image_over_pixel_limit() -> None:
    settings = InferenceSettings(INFERENCE_MAX_IMAGE_PIXELS=1, _env_file=None)

    with pytest.raises(ValueError, match=CLIENT_INPUT_ERROR):
        validate_image_bytes(_png_bytes(), settings)


def test_rejects_disallowed_image_format() -> None:
    settings = InferenceSettings(INFERENCE_ALLOWED_IMAGE_FORMATS="JPEG", _env_file=None)

    with pytest.raises(ValueError, match=CLIENT_INPUT_ERROR):
        validate_image_bytes(_png_bytes(), settings)


def test_rejects_pillow_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = InferenceSettings(INFERENCE_MAX_IMAGE_PIXELS=100, _env_file=None)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(ValueError, match=CLIENT_INPUT_ERROR):
        validate_image_bytes(_png_bytes(), settings)


# --- Parameter structure -----------------------------------------------------


@pytest.mark.parametrize(
    "setting,value",
    [
        ("INFERENCE_MAX_PARAMS_DEPTH", "1"),
        ("INFERENCE_MAX_TRANSCRIBE_LINES", "1"),
        ("INFERENCE_MAX_GEOMETRY_POINTS", "2"),
    ],
)
def test_rejects_excessive_parameter_structure(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
) -> None:
    monkeypatch.setenv(setting, value)
    params = {
        "INFERENCE_MAX_PARAMS_DEPTH": {"nested": {"too_deep": True}},
        "INFERENCE_MAX_TRANSCRIBE_LINES": {"lines": [{"line_index": 0}, {"line_index": 1}]},
        "INFERENCE_MAX_GEOMETRY_POINTS": {
            "lines": [{"line_index": 0, "points": [[0, 0], [1, 0], [1, 1]]}]
        },
    }[setting]

    with pytest.raises(ValueError, match=CLIENT_INPUT_ERROR):
        validate_request_params(params, InferenceSettings(_env_file=None))


def test_rejects_job_payload_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_MAX_JOB_PAYLOAD_BYTES", "1024")

    with pytest.raises(ValidationError, match=CLIENT_INPUT_ERROR):
        _segment_job(image_bytes=b"x" * 900, params={"padding": "x" * 200})


def test_accepts_thousands_of_full_page_transcribe_lines() -> None:
    lines = [
        {
            "line_id": str(uuid4()),
            "line_index": line_index,
            "points": [[point_index % 10, line_index % 10] for point_index in range(51)],
        }
        for line_index in range(2_000)
    ]

    request = _segment_job(
        task=InferenceTask.transcribe,
        registry_model_id="greek-calamari-v1",
        params={"lines": lines},
    )

    assert len(request.params["lines"]) == 2_000


# --- BLLA segmentation knobs -------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        # The decoder requires a threshold strictly between zero and one, and
        # the runtime clamps to 0.99.
        {"heatmap_threshold": 1.5},
        # Non-positive values are refused at the same seam.
        {"heatmap_threshold": 0},
        {"heatmap_threshold": -1},
    ],
)
def test_rejects_out_of_range_segment_params(params: dict) -> None:
    """Refused on the submission path, so no agent is ever handed the page."""
    with pytest.raises(ValidationError, match=CLIENT_INPUT_ERROR):
        _segment_job(params=params)


def test_accepts_segment_params_at_their_upper_bound() -> None:
    """The bound is inclusive, and an in-range request is a valid submission."""
    request = _segment_job(params={"heatmap_threshold": 0.99})

    assert request.params["heatmap_threshold"] == 0.99


def test_the_same_bounds_guard_the_runtime_the_agent_calls() -> None:
    """The agent runs whatever it claimed, so the seam cannot only be on submit.

    A page reaches ``run_model`` from a **claim**, not from a request body, and
    the platform is not the only thing that ever built one.
    """
    with pytest.raises(ValueError, match=CLIENT_INPUT_ERROR):
        validate_request_params({"heatmap_threshold": 10_000}, InferenceSettings(_env_file=None))
