"""POST /ml/v1/run integration tests — mock runner only."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from ml.api.app import create_app

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
    b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
MINIMAL_PNG_B64 = base64.b64encode(MINIMAL_PNG).decode()


@pytest.fixture
def ml_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ML_FORCE_MOCK_RUNNER", "1")
    return TestClient(create_app())


def test_run_transcribe_returns_contract_fields(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "greek-calamariv1",
            "registry_tag": "stable",
            "image_bytes": MINIMAL_PNG_B64,
            "params": {"line_index": 0},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    output = body["output"]
    assert output["text"] == "mock transcription 1"
    assert output["confidence"] == 0.91
    assert len(output["character_confidences"]) == len(output["text"])
    assert output["character_confidences"][0] == {"char": "m", "confidence": 0.91}


def test_run_transcribe_unknown_model_returns_404(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "missing-model",
            "image_bytes": MINIMAL_PNG_B64,
        },
    )

    assert response.status_code == 404


def test_run_segment_not_implemented(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "kraken-blla",
            "image_bytes": MINIMAL_PNG_B64,
        },
    )

    assert response.status_code == 501


def test_mock_runner_honors_line_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_FORCE_MOCK_RUNNER", "1")
    client = TestClient(create_app())
    response = client.post(
        "/ml/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "greek-calamariv1",
            "image_bytes": MINIMAL_PNG_B64,
            "params": {"line_index": 1},
        },
    )
    assert response.status_code == 200
    assert response.json()["output"]["text"] == "mock transcription 2"
    assert response.json()["output"]["confidence"] == 0.82
