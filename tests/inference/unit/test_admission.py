"""Focused admission-control tests that do not load inference weights."""

from __future__ import annotations

import base64
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from inference.admission import CLIENT_INPUT_ERROR, validate_image_bytes
from inference.api.app import create_app
from inference.contracts.common import InferenceTask
from inference.contracts.jobs import JobSubmitRequest
from inference.infrastructure.settings import InferenceSettings, get_inference_settings
from PIL import Image
from pydantic import ValidationError


def _png_bytes(*, size: tuple[int, int] = (2, 2)) -> bytes:
    output = BytesIO()
    Image.new("L", size).save(output, format="PNG")
    return output.getvalue()


def _payload(*, image_bytes: bytes, params: dict | None = None) -> dict:
    return {
        "task": InferenceTask.segment.value,
        "registry_model_id": "missing-model",
        "registry_tag": "stable",
        "image_bytes": base64.b64encode(image_bytes).decode(),
        "params": params or {},
    }


@pytest.fixture
def admission_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "admission-test-secret")

    def create_client() -> TestClient:
        get_inference_settings.cache_clear()
        return TestClient(
            create_app(),
            headers={"X-Inference-Service-Secret": "admission-test-secret"},
        )

    return create_client


def test_inference_service_routes_reject_missing_and_invalid_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "admission-test-secret")
    get_inference_settings.cache_clear()
    client = TestClient(create_app())

    assert client.post("/inference/v1/jobs", json={}).status_code == 401
    assert (
        client.post(
            "/inference/v1/jobs",
            json={},
            headers={"X-Inference-Service-Secret": "wrong-secret"},
        ).status_code
        == 403
    )


def test_unauthenticated_requests_do_not_exhaust_service_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "admission-test-secret")
    monkeypatch.setenv("INFERENCE_RATE_LIMIT_PER_MINUTE", "1")
    get_inference_settings.cache_clear()
    client = TestClient(create_app())

    assert (
        client.post(
            "/inference/v1/jobs",
            json={},
            headers={"X-Inference-Service-Secret": "admission-test-secret"},
        ).status_code
        == 422
    )
    assert client.post("/inference/v1/jobs", json={}).status_code == 401


def test_rejects_oversized_encoded_image_before_base64_decode(
    admission_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INFERENCE_MAX_ENCODED_IMAGE_BYTES", "1024")
    client = admission_client()

    response = client.post(
        "/inference/v1/run",
        json=_payload(image_bytes=b"x" * 800),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": CLIENT_INPUT_ERROR}


def test_rejects_oversized_decoded_image_before_image_loading(
    admission_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INFERENCE_MAX_DECODED_IMAGE_BYTES", "1024")
    client = admission_client()

    response = client.post(
        "/inference/v1/run",
        json=_payload(image_bytes=b"x" * 1025),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": CLIENT_INPUT_ERROR}


def test_rejects_oversized_request_body(admission_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_MAX_REQUEST_BODY_BYTES", "1024")
    client = admission_client()

    response = client.post(
        "/inference/v1/run",
        json=_payload(image_bytes=_png_bytes(), params={"padding": "x" * 2_000}),
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request exceeds configured limits"


def test_rejects_image_over_pixel_limit(admission_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_MAX_IMAGE_PIXELS", "1")
    client = admission_client()

    response = client.post("/inference/v1/run", json=_payload(image_bytes=_png_bytes()))

    assert response.status_code == 422
    assert response.json() == {"detail": CLIENT_INPUT_ERROR}


def test_rejects_disallowed_image_format(admission_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_ALLOWED_IMAGE_FORMATS", "JPEG")
    client = admission_client()

    response = client.post("/inference/v1/run", json=_payload(image_bytes=_png_bytes()))

    assert response.status_code == 422
    assert response.json() == {"detail": CLIENT_INPUT_ERROR}


def test_rejects_pillow_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = InferenceSettings(INFERENCE_MAX_IMAGE_PIXELS=100)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(ValueError, match=CLIENT_INPUT_ERROR):
        validate_image_bytes(_png_bytes(), settings)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("INFERENCE_MAX_PARAMS_DEPTH", "1"),
        ("INFERENCE_MAX_TRANSCRIBE_LINES", "1"),
        ("INFERENCE_MAX_GEOMETRY_POINTS", "2"),
    ],
)
def test_rejects_excessive_parameter_structure(
    admission_client,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
) -> None:
    monkeypatch.setenv(setting, value)
    client = admission_client()
    params = {
        "INFERENCE_MAX_PARAMS_DEPTH": {"nested": {"too_deep": True}},
        "INFERENCE_MAX_TRANSCRIBE_LINES": {"lines": [{"line_index": 0}, {"line_index": 1}]},
        "INFERENCE_MAX_GEOMETRY_POINTS": {
            "lines": [{"line_index": 0, "points": [[0, 0], [1, 0], [1, 1]]}]
        },
    }[setting]

    response = client.post(
        "/inference/v1/run",
        json=_payload(image_bytes=_png_bytes(), params=params),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": CLIENT_INPUT_ERROR}


def test_rejects_job_payload_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_MAX_JOB_PAYLOAD_BYTES", "1024")
    get_inference_settings.cache_clear()

    with pytest.raises(ValidationError, match=CLIENT_INPUT_ERROR):
        JobSubmitRequest(
            task=InferenceTask.segment,
            registry_model_id="missing-model",
            product_job_id=uuid4(),
            image_bytes=b"x" * 900,
            params={"padding": "x" * 200},
        )


def test_accepts_thousands_of_full_page_transcribe_lines() -> None:
    get_inference_settings.cache_clear()
    lines = [
        {
            "line_id": str(uuid4()),
            "line_index": line_index,
            "points": [[point_index % 10, line_index % 10] for point_index in range(51)],
        }
        for line_index in range(2_000)
    ]

    request = JobSubmitRequest(
        task=InferenceTask.transcribe,
        registry_model_id="greek-calamari-v1",
        product_job_id=uuid4(),
        image_bytes=_png_bytes(),
        params={"lines": lines},
    )

    assert len(request.params["lines"]) == 2_000


def test_worker_concurrency_configuration_is_bounded() -> None:
    assert InferenceSettings(INFERENCE_WORKER_CONCURRENCY=4).inference_worker_concurrency == 4
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        InferenceSettings(INFERENCE_WORKER_CONCURRENCY=0)
    with pytest.raises(ValidationError, match="less than or equal to 4"):
        InferenceSettings(INFERENCE_WORKER_CONCURRENCY=5)


@pytest.mark.parametrize(
    "params",
    [
        # A radius this large becomes a multi-thousand-pixel morphology kernel.
        {"use_otsu_refinement": True, "otsu_sphere_radius": 4_096},
        # Ratios that no mask comparison can ever satisfy.
        {"min_iou": 1.5},
        {"min_area_ratio": 12.0},
        # Past the stored-geometry cap the extra vertices are unusable.
        {"target_max_points": 100_000},
        {"split_vertical_gap_px": 100_000.0},
        # Non-positive values are refused at the same seam, matching the
        # platform's SegmentPartRequest bounds.
        {"otsu_sphere_radius": 0},
        {"min_iou": -1},
        {"target_max_points": 3},
    ],
)
def test_rejects_out_of_range_segment_params(admission_client, params: dict) -> None:
    client = admission_client()

    response = client.post(
        "/inference/v1/run",
        json=_payload(image_bytes=_png_bytes(), params=params),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": CLIENT_INPUT_ERROR}


def test_accepts_segment_params_at_their_upper_bound(admission_client) -> None:
    """The bound is inclusive, and an in-range request still reaches the runner."""
    client = admission_client()

    response = client.post(
        "/inference/v1/run",
        json=_payload(
            image_bytes=_png_bytes(),
            params={
                "use_otsu_refinement": True,
                "otsu_sphere_radius": 128,
                "min_iou": 1.0,
                "min_area_ratio": 2.0,
                "target_max_points": 256,
                "split_vertical_gap_px": 256,
            },
        ),
    )

    # 404: admission passed and the unknown registry model is what failed.
    assert response.status_code == 404


def test_out_of_range_segment_params_are_refused_on_the_job_path_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queued path never sees InferenceRunRequest, so it needs the same seam."""
    get_inference_settings.cache_clear()

    with pytest.raises(ValidationError, match=CLIENT_INPUT_ERROR):
        JobSubmitRequest(
            task=InferenceTask.segment,
            registry_model_id="blla-segment",
            product_job_id=uuid4(),
            image_bytes=_png_bytes(),
            params={"otsu_sphere_radius": 10_000},
        )
