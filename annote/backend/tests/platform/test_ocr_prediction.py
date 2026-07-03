"""Synchronous OCR prediction integration tests — public HTTP behavior."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask, ModelBinding
from backend.tests.platform.test_documents import MINIMAL_PNG
from backend.tests.platform.test_transcribe_pipeline import (
    _create_document_part_with_lines,
    _documents_url,
)
from infrastructure.db import SyncSessionLocal


@pytest.fixture(autouse=True)
def _reset_inference_tables() -> None:
    with SyncSessionLocal() as session:
        session.execute(delete(ModelBinding))
        session.execute(delete(InferenceModel))
        session.commit()
    yield
    with SyncSessionLocal() as session:
        session.execute(delete(ModelBinding))
        session.execute(delete(InferenceModel))
        session.commit()


def _seed_transcribe_model(*, name: str | None = None) -> InferenceModel:
    model_name = name or f"calamari-transcribe-{uuid.uuid4().hex[:8]}"
    with SyncSessionLocal() as session:
        model = InferenceModel(
            name=model_name,
            provider="calamari",
            task=InferenceTask.transcribe,
            artifact_ref="../model/checkpoints/best.ckpt",
            default_params={"device": "cpu"},
        )
        session.add(model)
        session.commit()
        session.refresh(model)
        session.expunge(model)
        return model


def test_segment_ocr_predict_writes_model_layer_and_preserves_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    model = _seed_transcribe_model()
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    line_id = lines.json()[0]["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/lines/{line_id}/ocr-predict",
        headers=owner_headers,
        json={"model_id": str(model.id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == str(model.id)
    assert body["lines"] == [
        {
            "line_id": line_id,
            "text": "mock transcription 1",
            "confidence": 0.91,
            "text_source": "model",
            "character_confidences": [
                {"char": char, "confidence": 0.91}
                for char in "mock transcription 1"
            ],
        }
    ]

    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers)
    assert layers.status_code == 200
    assert [layer["kind"] for layer in layers.json()] == ["ground_truth", "model"]

    refreshed = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert refreshed.status_code == 200
    first_line = refreshed.json()[0]
    kinds = [entry["transcription_kind"] for entry in first_line["line_transcriptions"]]
    assert "model" in kinds
    assert "ground_truth" not in kinds
    model_entry = next(
        entry for entry in first_line["line_transcriptions"] if entry["transcription_kind"] == "model"
    )
    assert model_entry["text_source"] == "model"
    assert model_entry["character_confidences"] is not None
    assert len(model_entry["character_confidences"]) == len(model_entry["text"])


def test_rerun_segment_ocr_overwrites_latest_model_layer(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    model = _seed_transcribe_model()
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    line_id = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers
    ).json()[0]["id"]

    first = client.post(
        f"{base}/{document_id}/parts/{part_id}/lines/{line_id}/ocr-predict",
        headers=owner_headers,
        json={"model_id": str(model.id)},
    )
    assert first.status_code == 200
    first_layer_id = first.json()["transcription_id"]

    second = client.post(
        f"{base}/{document_id}/parts/{part_id}/lines/{line_id}/ocr-predict",
        headers=owner_headers,
        json={"model_id": str(model.id)},
    )
    assert second.status_code == 200
    assert second.json()["transcription_id"] == first_layer_id

    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers)
    model_layers = [layer for layer in layers.json() if layer["kind"] == "model"]
    assert len(model_layers) == 1


def test_page_ocr_predict_writes_all_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    model = _seed_transcribe_model()
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/ocr-predict",
        headers=owner_headers,
        json={"model_id": str(model.id)},
    )

    assert response.status_code == 200
    assert [line["text"] for line in response.json()["lines"]] == [
        "mock transcription 1",
        "mock transcription 2",
    ]


def test_page_ocr_predict_rejects_empty_layout(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    model = _seed_transcribe_model()
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Empty codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/ocr-predict",
        headers=owner_headers,
        json={"model_id": str(model.id)},
    )

    assert response.status_code == 409
