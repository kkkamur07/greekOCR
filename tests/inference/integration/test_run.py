"""Tests for synchronous POST /inference/v1/run."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from tests.fixtures.paths import SEGMENT_PAGE, TRANSCRIBE_LINE

pytestmark = pytest.mark.integration


def _image_payload(path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# --- Sync segment run (ML lane) ---
# Tests POST /inference/v1/run with Kraken. Does not test async job queue.


@pytest.mark.ml
def test_run_segment_returns_kraken_layout(inference_client: TestClient) -> None:
    response = inference_client.post(
        "/inference/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "greek-kraken-segment-v1",
            "registry_tag": "stable",
            "image_bytes": _image_payload(SEGMENT_PAGE),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "segment"
    assert len(body["output"]["blocks"]) == 1
    assert len(body["output"]["lines"]) > 1
    assert body["output"]["lines"][0]["source_metadata"]["adapter"] == "kraken"


# --- Sync transcribe run (ML lane) ---
# Tests Calamari transcribe on a line crop. Does not test batch transcribe.


@pytest.mark.ml
def test_run_transcribe_returns_calamari_output(inference_client: TestClient) -> None:
    response = inference_client.post(
        "/inference/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "syriac-calamari-v1",
            "registry_tag": "stable",
            "image_bytes": _image_payload(TRANSCRIBE_LINE),
            "params": {"line_index": 0},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert isinstance(body["output"]["text"], str)
    assert 0.0 <= body["output"]["confidence"] <= 1.0


# --- Sync run validation ---
# Tests unknown registry model returns 404. Does not load weights or run inference.


def test_run_segment_unknown_model_returns_404(inference_client: TestClient) -> None:
    response = inference_client.post(
        "/inference/v1/run",
        json={
            "task": "segment",
            "registry_model_id": "missing-model",
            "registry_tag": "stable",
            "image_bytes": _image_payload(SEGMENT_PAGE),
        },
    )

    assert response.status_code == 404
