"""Job runner — enqueue, claim (SKIP LOCKED), worker loop, failure persistence."""

from __future__ import annotations

import infrastructure.models  # noqa: F401 — register all ORM mappers

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.jobs.infrastructure.job_repository import (
    claim_next_pending_job,
    count_active_jobs,
    lock_first_pending_job_for_test,
    mark_job_done,
)
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.jobs.infrastructure.worker import execute_claimed_job
from infrastructure.db import SyncSessionLocal


def _clear_all_jobs() -> None:
    with SyncSessionLocal() as session:
        session.execute(delete(Job))
        session.commit()


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
def _reset_jobs_table() -> None:
    """Isolate tests — real Postgres; hard delete before each test; drain worker after."""
    _clear_all_jobs()
    yield
    _wait_until_no_active_test_jobs(timeout=5.0)
    _clear_all_jobs()


def _poll_job(
    client: TestClient,
    job_id: str,
    *,
    expect: str,
    headers: dict[str, str],
    timeout: float = 5.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        body = response.json()
        if body["status"] == expect:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status {expect!r} in {timeout}s")


def test_enqueue_test_job_requires_auth(client: TestClient):
    response = client.post("/jobs/test")
    assert response.status_code == 401


def test_get_job_requires_auth(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 401


def test_enqueue_test_job_returns_job_id_immediately(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", headers=auth_headers)

    assert response.status_code == 201
    job_id = response.json()["job_id"]
    uuid.UUID(job_id)

    body = _poll_job(client, job_id, expect="done", headers=auth_headers)
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


def test_get_job_denies_access_to_other_users_job(client: TestClient):
    owner_suffix = uuid.uuid4().hex[:8]
    other_suffix = uuid.uuid4().hex[:8]

    def _register(suffix: str) -> str:
        r = client.post(
            "/auth/register",
            json={"email": f"{suffix}@test.kalamos", "username": suffix, "password": "test-pass-123"},
        )
        return r.json()["access_token"]

    owner_token = _register(f"owner-{owner_suffix}")
    other_token = _register(f"other-{other_suffix}")

    created = client.post("/jobs/test", headers={"Authorization": f"Bearer {owner_token}"})
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}", headers={"Authorization": f"Bearer {other_token}"})
    assert response.status_code == 403


def test_get_job_denies_access_to_ownerless_job(client: TestClient, registered_user: dict[str, str]):
    job_id = uuid.uuid4()
    with SyncSessionLocal() as session:
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


def test_claim_next_uses_skip_locked():
    """While one transaction holds FOR UPDATE on job A, SKIP LOCKED claims job B."""
    job_a_id = uuid.uuid4()
    job_b_id = uuid.uuid4()
    with SyncSessionLocal() as session:
        session.add_all(
            [
                Job(
                    id=job_a_id,
                    type=JobType.pipeline,
                    status=JobStatus.pending,
                    payload={"handler": "noop", "order": 0},
                ),
                Job(
                    id=job_b_id,
                    type=JobType.pipeline,
                    status=JobStatus.pending,
                    payload={"handler": "noop", "order": 1},
                ),
            ]
        )
        session.commit()

    hold_session, locked = lock_first_pending_job_for_test()
    assert locked is not None
    assert locked.id == job_a_id
    try:
        second = claim_next_pending_job()
        assert second is not None
        assert second.id == job_b_id
        assert second.status == JobStatus.running
        mark_job_done(second.id)
    finally:
        hold_session.rollback()
        hold_session.close()

    first = claim_next_pending_job()
    assert first is not None
    assert first.id == job_a_id
    mark_job_done(first.id)


def test_concurrent_claimers_do_not_run_same_job_twice():
    """Two threads claiming in parallel must not claim the same row."""
    job_ids = [uuid.uuid4() for _ in range(4)]
    with SyncSessionLocal() as session:
        session.add_all(
            [
                Job(
                    id=jid,
                    type=JobType.pipeline,
                    status=JobStatus.pending,
                    payload={"handler": "noop"},
                )
                for jid in job_ids
            ]
        )
        session.commit()

    claimed: list[uuid.UUID] = []
    lock = threading.Lock()

    def claim_once() -> None:
        job = claim_next_pending_job()
        if job is not None:
            with lock:
                claimed.append(job.id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(claim_once) for _ in range(8)]
        for fut in as_completed(futures):
            fut.result()

    assert len(claimed) == len(set(claimed))
    assert set(claimed).issubset(set(job_ids))
    assert len(claimed) == 4

    with SyncSessionLocal() as session:
        running = (
            session.execute(
                select(Job).where(Job.id.in_(job_ids), Job.status == JobStatus.running)
            )
            .scalars()
            .all()
        )
        assert len(running) == 4

    for jid in job_ids:
        mark_job_done(jid)


def test_failed_handler_stores_error_message(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", json={"handler": "fail"}, headers=auth_headers)
    job_id = response.json()["job_id"]

    body = _poll_job(client, job_id, expect="failed", headers=auth_headers)
    assert body["error"] == "intentional test failure"
    assert body["completed_at"] is not None


def test_worker_processes_pending_job_via_lifespan(client: TestClient, registered_user: dict[str, str]):
    """Background loop started in app lifespan completes a noop job without manual claim."""
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", headers=auth_headers)
    job_id = response.json()["job_id"]
    body = _poll_job(client, job_id, expect="done", headers=auth_headers, timeout=8.0)
    assert body["result"] == {"ok": True}


def test_execute_claimed_job_marks_unknown_handler_failed():
    job_id = uuid.uuid4()
    with SyncSessionLocal() as session:
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

    with SyncSessionLocal() as session:
        row = session.get(Job, job_id)
        assert row is not None
        assert row.status == JobStatus.failed
        assert "no handler" in (row.error or "").lower()
