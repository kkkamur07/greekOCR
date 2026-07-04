"""Tests for synchronous POST /ml/v1/run."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from ml.api.app import create_app

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
    b"\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def ml_client() -> TestClient:
    return TestClient(create_app())


def test_run_segment_returns_mock_layout(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "kraken-blla",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(MINIMAL_PNG).decode(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "segment"
    assert len(body["output"]["blocks"]) == 1
    assert len(body["output"]["lines"]) == 1


def test_run_transcribe_returns_mock_text(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "greek-calamariv1",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(MINIMAL_PNG).decode(),
            "params": {"line_index": 1},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["output"]["text"] == "mock transcription 2"
    assert body["output"]["confidence"] == 0.82


def test_run_segment_unknown_model_returns_404(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "missing-model",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(MINIMAL_PNG).decode(),
        },
    )

    assert response.status_code == 404
