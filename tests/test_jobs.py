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

from backend.inference.infrastructure.job_repository import claim_next_pending_job, mark_job_done
from backend.inference.infrastructure.orm_models import Job, JobStatus, JobType
from backend.inference.infrastructure.worker import execute_claimed_job
from infrastructure.db import SyncSessionLocal


@pytest.fixture(autouse=True)
def _reset_jobs_table() -> None:
    """Isolate tests — background worker and claim tests share the jobs table."""
    with SyncSessionLocal() as session:
        session.execute(delete(Job))
        session.commit()
    yield
    time.sleep(0.15)
    with SyncSessionLocal() as session:
        session.execute(delete(Job))
        session.commit()


def _poll_job(client: TestClient, job_id: str, *, expect: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] == expect:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status {expect!r} in {timeout}s")


def test_enqueue_test_job_returns_job_id_immediately(client: TestClient):
    response = client.post("/jobs/test")

    assert response.status_code == 201
    job_id = response.json()["job_id"]
    uuid.UUID(job_id)

    body = _poll_job(client, job_id, expect="done")
    assert body["type"] == "pipeline"
    assert body["payload"]["handler"] == "noop"
    assert body["error"] is None
    assert body["started_at"] is not None
    assert body["completed_at"] is not None


def test_get_job_returns_status_and_timestamps(client: TestClient):
    created = client.post("/jobs/test")
    job_id = created.json()["job_id"]

    pending = client.get(f"/jobs/{job_id}")
    assert pending.status_code == 200
    assert pending.json()["status"] in ("pending", "running", "done")

    missing = client.get(f"/jobs/{uuid.uuid4()}")
    assert missing.status_code == 404


def test_claim_next_uses_skip_locked():
    """Direct claim: second claimer gets a different job while first holds running."""
    job_a_id = uuid.uuid4()
    job_b_id = uuid.uuid4()
    with SyncSessionLocal() as session:
        session.add_all(
            [
                Job(
                    id=job_a_id,
                    type=JobType.pipeline,
                    status=JobStatus.pending,
                    payload={"handler": "noop"},
                ),
                Job(
                    id=job_b_id,
                    type=JobType.pipeline,
                    status=JobStatus.pending,
                    payload={"handler": "noop"},
                ),
            ]
        )
        session.commit()

    first = claim_next_pending_job()
    second = claim_next_pending_job()
    assert first is not None
    assert second is not None
    assert first.id != second.id
    assert first.status == JobStatus.running
    assert second.status == JobStatus.running

    mark_job_done(first.id)
    mark_job_done(second.id)


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


def test_failed_handler_stores_error_message(client: TestClient):
    response = client.post("/jobs/test", json={"handler": "fail"})
    job_id = response.json()["job_id"]

    body = _poll_job(client, job_id, expect="failed")
    assert body["error"] == "intentional test failure"
    assert body["completed_at"] is not None


def test_worker_processes_pending_job_via_lifespan(client: TestClient):
    """Background loop started in app lifespan completes a noop job without manual claim."""
    response = client.post("/jobs/test")
    job_id = response.json()["job_id"]
    body = _poll_job(client, job_id, expect="done", timeout=8.0)
    assert body["result"] == {"ok": True}


def test_execute_claimed_job_marks_unknown_handler_failed():
    job_id = uuid.uuid4()
    with SyncSessionLocal() as session:
        job = Job(
            id=job_id,
            type=JobType.segment,
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
        assert "no handler" in (row.error or "")
