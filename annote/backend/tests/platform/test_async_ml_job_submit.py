"""Async ML job submit — worker delegates to POST /ml/v1/jobs and callbacks complete jobs."""

from __future__ import annotations

import uuid
from io import BytesIO

from backend.jobs.api.dependencies import ML_WEBHOOK_SECRET_HEADER
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from backend.tests.platform.ml_test_helpers import (
    drive_platform_worker_once,
    list_ml_jobs,
    poll_platform_job,
    wait_for_ml_jobs,
)
from fastapi.testclient import TestClient
from ml_service.contracts.common import MLJobStatus, MLTask
from ml_service.contracts.jobs import JobCallbackRequest
from PIL import Image

import pytest

from infrastructure.db import SyncSessionLocal


def _segment_test_png() -> bytes:
    """Kraken needs a non-trivial page image; 1×1 px yields zero regions."""
    buffer = BytesIO()
    Image.new("RGB", (128, 128), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_worker_submits_segment_job_to_ml_api(
    client: TestClient,
    paused_ml_worker,
    owner_headers: dict[str, str],
    owner_project: dict,
):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Submit probe"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _segment_test_png(), "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={},
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["job_id"])

    drive_platform_worker_once(job_id)
    job = poll_platform_job(job_id, expect=JobStatus.waiting, timeout=10.0)
    assert job.ml_job_id is not None
    ml_jobs = list_ml_jobs(product_job_id=job_id)
    assert len(ml_jobs) == 1
    assert ml_jobs[0].task == "segment"
    assert ml_jobs[0].product_job_id == job_id


def test_ml_submit_failure_marks_job_failed(
    client: TestClient,
    broken_ml_service,
    owner_headers: dict[str, str],
    owner_project: dict,
):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Submit failure"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _segment_test_png(), "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={},
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["job_id"])

    drive_platform_worker_once(job_id)
    job = poll_platform_job(job_id, expect=JobStatus.failed, timeout=10.0)
    assert job.error == "Job failed"


@pytest.mark.integration
def test_segment_submit_and_callback_marks_job_done(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Async segment"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _segment_test_png(), "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={},
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["job_id"])

    job = poll_platform_job(job_id, expect=JobStatus.done, timeout=120.0)
    assert job.result is not None
    assert job.result["blocks_count"] >= 0
    assert job.result["lines_count"] >= 0


@pytest.mark.integration
def test_transcribe_submit_and_callbacks_merge_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Async transcribe"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _segment_test_png(), "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]
    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "kraken",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "kraken",
                },
            ]
        },
    )
    assert replace.status_code == 200

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["job_id"])

    job = poll_platform_job(job_id, expect=JobStatus.done, timeout=300.0)
    assert job.result is not None
    assert len(job.result["lines"]) == 2
    for line in job.result["lines"]:
        assert isinstance(line["text"], str)
        assert line["text"].strip()


def test_transcribe_submit_uses_selected_registry_model(
    client: TestClient,
    paused_ml_worker,
    owner_headers: dict[str, str],
    owner_project: dict,
):
    with SyncSessionLocal() as session:
        model = InferenceModel(
            name=f"syriac-calamariv1-{uuid.uuid4().hex[:8]}",
            provider="calamari",
            task=InferenceTask.transcribe,
            artifact_ref="registry://syriac-calamariv1?tag=stable",
            default_params={"device": "cpu", "language": "grc"},
        )
        session.add(model)
        session.commit()
        model_id = model.id

    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Selected transcribe"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _segment_test_png(), "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]
    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "kraken",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "kraken",
                },
            ]
        },
    )
    assert replace.status_code == 200

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
        json={"model_id": str(model_id)},
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["job_id"])

    ml_jobs = wait_for_ml_jobs(product_job_id=job_id, count=2, timeout=20.0)
    with SyncSessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.model_id == model_id

    assert {entry.registry_model_id for entry in ml_jobs} == {"syriac-calamariv1"}
    assert {entry.registry_tag for entry in ml_jobs} == {"stable"}
    assert {entry.params.get("language") for entry in ml_jobs} == {"grc"}


def test_callback_missing_webhook_secret_returns_401(client: TestClient):
    job_id = uuid.uuid4()
    ml_job_id = uuid.uuid4()
    with SyncSessionLocal() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.segment,
                status=JobStatus.waiting,
                payload={},
                ml_job_id=ml_job_id,
            )
        )
        session.commit()

    callback = JobCallbackRequest(
        ml_job_id=ml_job_id,
        product_job_id=job_id,
        task=MLTask.segment,
        status=MLJobStatus.failed,
        error="boom",
    )
    response = client.post(
        "/internal/ml/job-complete",
        json=callback.model_dump(mode="json"),
    )
    assert response.status_code == 401

    response = client.post(
        "/internal/ml/job-complete",
        headers={ML_WEBHOOK_SECRET_HEADER: "wrong"},
        json=callback.model_dump(mode="json"),
    )
    assert response.status_code == 403
