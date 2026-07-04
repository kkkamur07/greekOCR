"""Tests for synchronous POST /ml/v1/run."""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from ml.architectures.mock import mock_segment
from PIL import Image

from ml.api.app import create_app


def _png(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def ml_client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def synthetic_segment_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _run_segment(*, image_bytes: bytes, **_kwargs):
        return mock_segment(image_bytes)

    monkeypatch.setattr("ml.api.run.run_segment", _run_segment)


def test_run_segment_returns_layout(
    ml_client: TestClient, synthetic_segment_runner: None
) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "kraken-blla",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(_png(10, 20)).decode(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "segment"
    output = body["output"]
    assert output["blocks"] == [
        {
            "external_id": "kraken-block-1",
            "order": 0,
            "box": {"points": [[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]]},
        }
    ]
    assert output["lines"][0]["external_id"] == "kraken-line-1"
    assert output["lines"][0]["block_external_id"] == "kraken-block-1"
    assert output["lines"][0]["points"] == [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 8.0],
        [0.0, 8.0],
    ]
    assert output["lines"][0]["baseline"] == {"points": output["lines"][0]["points"]}
    assert output["lines"][0]["mask"] == {"points": output["lines"][0]["points"]}
    assert output["lines"][0]["source_metadata"] == {"adapter": "kraken_stub"}


def test_run_segment_unknown_model_returns_404(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "missing-model",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(_png(1, 1)).decode(),
        },
    )

    assert response.status_code == 404


def test_run_transcribe_task_returns_422(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "greek-calamariv1",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(_png(1, 1)).decode(),
        },
    )

    assert response.status_code == 422
