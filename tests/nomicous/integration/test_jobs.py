"""Product job runner — enqueue, status, worker loop, failure persistence."""

from __future__ import annotations

import infrastructure.models  # noqa: F401 — register all ORM mappers

import time
import uuid

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from inference.contracts.jobs import JobSubmitRequest, JobSubmitResponse

from sqlalchemy import update

from backend.jobs.infrastructure.job_repository import (
    claim_next_pending_job,
    count_active_jobs,
    reclaim_stale_running_jobs,
)
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.jobs.infrastructure.worker import execute_claimed_job
from infrastructure.db import sync_system_session
from tests.nomicous.integration.helpers import (
    MINIMAL_PNG,
    documents_url,
    poll_job,
)


def _wait_until_no_active_test_jobs(*, timeout: float = 5.0) -> None:
    """Wait until the lifespan worker finished API test jobs (payload test=true)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if count_active_jobs(test_payload=True) == 0:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"test jobs still active after {timeout}s (count={count_active_jobs(test_payload=True)})"
    )


@pytest.fixture(autouse=True)
def _drain_active_test_jobs() -> None:
    """Let the lifespan worker finish API test jobs before parent conftest truncates."""
    yield
    _wait_until_no_active_test_jobs(timeout=5.0)


def _create_document_part_with_lines(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
) -> tuple[str, str, list[dict]]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Transcribe batch job"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
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
                    "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "source": "manual",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 1], [1, 1], [1, 2], [0, 2]],
                    "source": "manual",
                },
            ]
        },
    )
    assert replace.status_code == 200
    return document_id, part_id, replace.json()


class _CapturingInferenceClient:
    def __init__(self) -> None:
        self.requests: list[JobSubmitRequest] = []
        self.inference_job_id = uuid.uuid4()

    def submit_job(self, request: JobSubmitRequest) -> JobSubmitResponse:
        self.requests.append(request)
        return JobSubmitResponse(inference_job_id=self.inference_job_id)


# --- Jobs API auth ---
# Tests /jobs endpoints require a logged-in user. Does not test job execution.


def test_enqueue_test_job_requires_auth(client: TestClient):
    response = client.post("/jobs/test")
    assert response.status_code == 401


def test_get_job_requires_auth(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 401


def test_job_events_requires_auth(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}/events")
    assert response.status_code == 401


# --- Test job lifecycle ---
# Tests enqueue, poll, and status fields for noop handler. Does not test real ML jobs.


def test_enqueue_test_job_returns_job_id_immediately(
    client: TestClient, registered_user: dict[str, str]
):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", headers=auth_headers)

    assert response.status_code == 201
    job_id = response.json()["job_id"]
    uuid.UUID(job_id)

    body = poll_job(client, job_id, expect_status="done", headers=auth_headers)
    assert body["type"] == "pipeline"
    assert body["payload"]["handler"] == "noop"
    assert body["error"] is None
    assert body["started_at"] is not None
    assert body["completed_at"] is not None


def test_get_job_returns_status_and_timestamps(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]

    pending = client.get(f"/jobs/{job_id}", headers=auth_headers)
    assert pending.status_code == 200
    assert pending.json()["status"] in ("pending", "running", "done")

    missing = client.get(f"/jobs/{uuid.uuid4()}", headers=auth_headers)
    assert missing.status_code == 404


def test_cancel_pending_job_marks_cancelled_and_discards_partials(
    client: TestClient, registered_user: dict[str, str]
):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    me = client.get("/me", headers=auth_headers)
    assert me.status_code == 200
    user_id = uuid.UUID(me.json()["id"])

    # Keep the job from completing before we cancel: insert directly as pending
    # without relying on the worker race.
    with sync_system_session() as session:
        job = Job(
            type=JobType.pipeline,
            status=JobStatus.pending,
            payload={"handler": "noop", "test": True},
            user_id=user_id,
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = str(job.id)

    cancelled = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers)
    assert cancelled.status_code == 200, cancelled.text
    body = cancelled.json()
    assert body["status"] == "cancelled"
    assert body["result"] is None
    assert body["completed_at"] is not None

    again = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers)
    assert again.status_code == 409


def test_job_events_streams_current_snapshot(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]
    poll_job(client, job_id, expect_status="done", headers=auth_headers)

    with client.stream("GET", f"/jobs/{job_id}/events", headers=auth_headers) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: job" in body
    assert f'"id":"{job_id}"' in body
    assert '"status":"done"' in body


# --- Job access control ---
# Tests users can only read their own jobs. Does not test admin or shared-project access.


def test_get_job_denies_access_to_other_users_job(client: TestClient):
    owner_suffix = uuid.uuid4().hex[:8]
    other_suffix = uuid.uuid4().hex[:8]

    def _register(suffix: str) -> str:
        r = client.post(
            "/auth/register",
            json={
                "email": f"{suffix}@test.kalamos",
                "username": suffix,
                "password": "test-pass-123",
            },
        )
        return r.json()["access_token"]

    owner_token = _register(f"owner-{owner_suffix}")
    other_token = _register(f"other-{other_suffix}")

    created = client.post("/jobs/test", headers={"Authorization": f"Bearer {owner_token}"})
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}", headers={"Authorization": f"Bearer {other_token}"})
    assert response.status_code == 403


def test_get_job_denies_access_to_ownerless_job(
    client: TestClient, registered_user: dict[str, str]
):
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.pipeline,
                status=JobStatus.done,
                payload={"handler": "noop"},
                result={"ok": True},
                user_id=None,
            )
        )
        session.commit()

    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.get(f"/jobs/{job_id}", headers=auth_headers)

    assert response.status_code == 403


# --- Job failure and lifespan worker ---
# Tests error persistence and background worker loop. Does not test inference callbacks.


def test_failed_handler_stores_error_message(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", json={"handler": "fail"}, headers=auth_headers)
    job_id = response.json()["job_id"]

    body = poll_job(client, job_id, expect_status="failed", headers=auth_headers)
    assert body["error"] == "Test job failed"
    assert body["completed_at"] is not None


def test_worker_processes_pending_job_via_lifespan(
    client: TestClient, registered_user: dict[str, str]
):
    """Background loop started in app lifespan completes a noop job without manual claim."""
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", headers=auth_headers)
    job_id = response.json()["job_id"]
    body = poll_job(client, job_id, expect_status="done", headers=auth_headers, timeout=8.0)
    assert body["result"] == {"ok": True}


# --- Transcribe enqueue (stubbed inference client) ---
# Tests batched line payload sent to inference. Does not run Kraken or Calamari.


def test_transcribe_job_submits_one_batched_inference_job(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    from backend.jobs.infrastructure import worker as worker_module

    document_id, part_id, lines = _create_document_part_with_lines(
        client,
        owner_headers,
        owner_project,
    )
    stub_client = _CapturingInferenceClient()
    monkeypatch.setattr(worker_module, "_get_inference_client", lambda: stub_client)

    base = documents_url(owner_project["id"])
    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert response.status_code == 202

    body = poll_job(
        client,
        response.json()["job_id"],
        expect_status="waiting",
        headers=owner_headers,
        timeout=8.0,
    )
    with sync_system_session() as session:
        job = session.get(Job, uuid.UUID(body["id"]))
        assert job is not None
        assert job.inference_job_id == stub_client.inference_job_id
    assert len(stub_client.requests) == 1
    request = stub_client.requests[0]
    assert request.task.value == "transcribe"
    assert [line["line_id"] for line in request.params["lines"]] == [line["id"] for line in lines]
    assert [line["line_index"] for line in request.params["lines"]] == [0, 1]


# --- Handler dispatch ---
# Tests unknown job types fail cleanly. Does not test registered handlers.


def test_reclaim_stale_running_jobs_moves_expired_jobs_to_pending():
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.pipeline,
                status=JobStatus.pending,
                payload={"handler": "noop", "test": True},
            )
        )
        session.commit()

    claimed = claim_next_pending_job(test_only=True)
    assert claimed is not None
    assert claimed.id == job_id

    with sync_system_session() as session:
        running = session.get(Job, job_id)
        assert running is not None
        running.started_at = datetime.now(UTC) - timedelta(minutes=31)
        session.commit()

    reclaimed = reclaim_stale_running_jobs(running_timeout_seconds=1800.0)
    assert reclaimed == 1

    with sync_system_session() as session:
        recovered = session.get(Job, job_id)
        assert recovered is not None
        assert recovered.status == JobStatus.pending
        assert recovered.started_at is None

    with sync_system_session() as session:
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status=JobStatus.failed,
                completed_at=datetime.now(UTC),
            )
        )
        session.commit()


def test_execute_claimed_job_marks_unknown_handler_failed():
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        job = Job(
            id=job_id,
            type=JobType.binarize,
            status=JobStatus.running,
            payload={"test": False},
        )
        session.add(job)
        session.commit()
        session.refresh(job)

    execute_claimed_job(job)

    with sync_system_session() as session:
        row = session.get(Job, job_id)
        assert row is not None
        assert row.status == JobStatus.failed
        assert (
            "not supported" in (row.error or "").lower()
            or "no handler" in (row.error or "").lower()
        )
