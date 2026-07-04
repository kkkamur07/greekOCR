"""Async ML job submit — worker delegates to POST /ml/v1/jobs and callbacks complete jobs."""

from __future__ import annotations

import time
import uuid
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.jobs.api.dependencies import ML_WEBHOOK_SECRET_HEADER
from backend.jobs.infrastructure import worker as worker_module
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.tests.platform.conftest import install_ml_jobs_mock
from ml.contracts.common import MLJobStatus, MLTask
from ml.contracts.jobs import JobCallbackRequest

from infrastructure.db import SyncSessionLocal


def _one_pixel_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


def _poll_job_status(job_id: uuid.UUID, *, expect: JobStatus, timeout: float = 3.0) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with SyncSessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            if job.status == expect:
                session.expunge(job)
                return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {expect.value}")


def _drive_worker_once(job_id: uuid.UUID, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with SyncSessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            if job.status != JobStatus.pending:
                return
        worker_module.process_one_job()
        time.sleep(0.02)
    raise AssertionError(f"worker did not claim job {job_id}")


def test_worker_submits_segment_job_to_ml_api(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
):
    ml_mock = install_ml_jobs_mock(client, auto_callback=False)
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Submit probe"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _one_pixel_png(), "image/png")},
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

    _drive_worker_once(job_id)
    job = _poll_job_status(job_id, expect=JobStatus.waiting)
    assert job.ml_job_id is not None
    assert len(ml_mock.submitted) == 1
    assert ml_mock.submitted[0]["task"] == "segment"
    assert ml_mock.submitted[0]["product_job_id"] == str(job_id)


def test_ml_submit_failure_marks_job_failed(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
):
    install_ml_jobs_mock(client, auto_callback=False, fail_submit=True)
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Submit failure"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _one_pixel_png(), "image/png")},
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

    _drive_worker_once(job_id)
    job = _poll_job_status(job_id, expect=JobStatus.failed, timeout=5.0)
    assert job.error == "Job failed"


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
        files={"file": ("folio.png", _one_pixel_png(), "image/png")},
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

    _drive_worker_once(job_id)
    job = _poll_job_status(job_id, expect=JobStatus.done, timeout=5.0)
    assert job.result is not None
    assert job.result["blocks_count"] == 1
    assert job.result["lines_count"] == 1


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
        files={"file": ("folio.png", _one_pixel_png(), "image/png")},
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

    _drive_worker_once(job_id)
    job = _poll_job_status(job_id, expect=JobStatus.done, timeout=8.0)
    assert job.result is not None
    assert [line["text"] for line in job.result["lines"]] == [
        "mock transcription 1",
        "mock transcription 2",
    ]


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
