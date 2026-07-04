"""Tests for synchronous POST /ml/v1/run."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ml_service.api.app import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENT_IMAGE_PATH = (
    REPO_ROOT
    / "annote/data/manuscripts/pages/Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_6.jpeg"
)
TRANSCRIBE_IMAGE_PATH = (
    REPO_ROOT
    / "annote/data/manuscripts/export/Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_6_1.jpg"
)


@pytest.fixture
def ml_client() -> TestClient:
    return TestClient(create_app())


def _image_payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def test_run_segment_returns_kraken_layout(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "kraken-blla",
            "registry_tag": "stable",
            "image_bytes": _image_payload(SEGMENT_IMAGE_PATH),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "segment"
    assert len(body["output"]["blocks"]) == 1
    assert len(body["output"]["lines"]) > 1
    assert body["output"]["lines"][0]["source_metadata"]["adapter"] == "kraken"


def test_run_transcribe_returns_calamari_output(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "syriac-calamariv1",
            "registry_tag": "stable",
            "image_bytes": _image_payload(TRANSCRIBE_IMAGE_PATH),
            "params": {"line_index": 0},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert isinstance(body["output"]["text"], str)
    assert 0.0 <= body["output"]["confidence"] <= 1.0


def test_run_segment_unknown_model_returns_404(ml_client: TestClient) -> None:
    response = ml_client.post(
        "/ml/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "missing-model",
            "registry_tag": "stable",
            "image_bytes": _image_payload(SEGMENT_IMAGE_PATH),
        },
    )

    assert response.status_code == 404
