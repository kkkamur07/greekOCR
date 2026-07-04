"""ML catalog and binding integration tests — real Postgres (kalamos)."""

import uuid

import pytest
from sqlalchemy import delete

from backend.ml.infrastructure.orm_models import (
    InferenceModel,
    InferenceTask,
    ModelBinding,
)
from infrastructure.db import SyncSessionLocal

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
    b"\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe"
    b"\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


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


def _seed_model(*, name: str, task: InferenceTask) -> InferenceModel:
    with SyncSessionLocal() as session:
        model = InferenceModel(
            name=name,
            provider="kraken",
            task=task,
            artifact_ref=f"model/kraken/{name}.mlmodel",
            default_params={"device": "cpu"},
        )
        session.add(model)
        session.commit()
        session.refresh(model)
        session.expunge(model)
        return model


def _create_project_document_part(client, headers) -> tuple[str, str, str]:
    project = client.post(
        "/projects",
        headers=headers,
        json={"slug": f"models-{uuid.uuid4().hex[:8]}", "name": "Model tests"},
    )
    assert project.status_code == 201
    project_id = project.json()["id"]

    document = client.post(
        f"/projects/{project_id}/documents",
        headers=headers,
        json={"name": "Vat. gr. 1"},
    )
    assert document.status_code == 201
    document_id = document.json()["id"]

    part = client.post(
        f"/projects/{project_id}/documents/{document_id}/parts",
        headers=headers,
        files={"file": ("folio.png", MINIMAL_PNG, "image/png")},
    )
    assert part.status_code == 201
    return project_id, document_id, part.json()["id"]


@pytest.mark.integration
def test_list_inference_models_returns_catalog_entries(client, auth_headers):
    segment_model = _seed_model(
        name=f"kraken-segment-{uuid.uuid4().hex[:8]}",
        task=InferenceTask.segment,
    )
    transcribe_model = _seed_model(
        name=f"kraken-transcribe-{uuid.uuid4().hex[:8]}",
        task=InferenceTask.transcribe,
    )

    response = client.get("/inference/models", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    by_id = {item["id"]: item for item in body}
    assert by_id[str(segment_model.id)]["task"] == "segment"
    assert by_id[str(segment_model.id)]["default_params"] == {"device": "cpu"}
    assert by_id[str(transcribe_model.id)]["task"] == "transcribe"


@pytest.mark.integration
def test_part_binding_overrides_document_overrides_project(client, auth_headers):
    project_model = _seed_model(
        name=f"kraken-project-{uuid.uuid4().hex[:8]}",
        task=InferenceTask.segment,
    )
    document_model = _seed_model(
        name=f"kraken-document-{uuid.uuid4().hex[:8]}",
        task=InferenceTask.segment,
    )
    part_model = _seed_model(
        name=f"kraken-part-{uuid.uuid4().hex[:8]}",
        task=InferenceTask.segment,
    )
    project_id, document_id, part_id = _create_project_document_part(client, auth_headers)

    project_binding = client.post(
        f"/projects/{project_id}/model-bindings",
        headers=auth_headers,
        json={
            "task": "segment",
            "model_id": str(project_model.id),
            "overrides": {"level": "project"},
        },
    )
    assert project_binding.status_code == 201

    document_binding = client.post(
        f"/projects/{project_id}/documents/{document_id}/model-bindings",
        headers=auth_headers,
        json={
            "task": "segment",
            "model_id": str(document_model.id),
            "overrides": {"level": "document"},
        },
    )
    assert document_binding.status_code == 201

    part_binding = client.post(
        f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/model-bindings",
        headers=auth_headers,
        json={
            "task": "segment",
            "model_id": str(part_model.id),
            "overrides": {"level": "part"},
        },
    )
    assert part_binding.status_code == 201

    resolved = client.get(
        (
            f"/projects/{project_id}/documents/{document_id}/parts/{part_id}"
            "/model-bindings/resolve?task=segment"
        ),
        headers=auth_headers,
    )
    assert resolved.status_code == 200
    assert resolved.json()["model"]["id"] == str(part_model.id)
    assert resolved.json()["effective_params"]["level"] == "part"

    delete_part = client.delete(
        f"/projects/{project_id}/model-bindings/{part_binding.json()['id']}",
        headers=auth_headers,
    )
    assert delete_part.status_code == 204
    resolved_after_part_delete = client.get(
        (
            f"/projects/{project_id}/documents/{document_id}/parts/{part_id}"
            "/model-bindings/resolve?task=segment"
        ),
        headers=auth_headers,
    )
    assert resolved_after_part_delete.status_code == 200
    assert resolved_after_part_delete.json()["model"]["id"] == str(document_model.id)

    update_document = client.patch(
        f"/projects/{project_id}/model-bindings/{document_binding.json()['id']}",
        headers=auth_headers,
        json={"overrides": {"level": "document-updated"}},
    )
    assert update_document.status_code == 200
    assert update_document.json()["overrides"] == {"level": "document-updated"}


@pytest.mark.integration
def test_transcribe_resolver_returns_project_binding(client, auth_headers):
    transcribe_model = _seed_model(
        name=f"kraken-transcribe-{uuid.uuid4().hex[:8]}",
        task=InferenceTask.transcribe,
    )
    project_id, document_id, part_id = _create_project_document_part(client, auth_headers)

    binding = client.post(
        f"/projects/{project_id}/model-bindings",
        headers=auth_headers,
        json={
            "task": "transcribe",
            "model_id": str(transcribe_model.id),
            "overrides": {"language": "grc"},
        },
    )
    assert binding.status_code == 201

    resolved = client.get(
        (
            f"/projects/{project_id}/documents/{document_id}/parts/{part_id}"
            "/model-bindings/resolve?task=transcribe"
        ),
        headers=auth_headers,
    )

    assert resolved.status_code == 200
    assert resolved.json()["model"]["id"] == str(transcribe_model.id)
    assert resolved.json()["effective_params"]["language"] == "grc"
