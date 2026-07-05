"""Helpers for platform tests that drive the real ML job queue."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from backend.jobs.infrastructure import worker as worker_module
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from infrastructure.db import SyncSessionLocal

if TYPE_CHECKING:
    from ml_service.infrastructure.orm_models import MLJob


def poll_platform_job(job_id: uuid.UUID, *, expect: JobStatus, timeout: float = 30.0) -> Job:
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


def drive_platform_worker_once(job_id: uuid.UUID, *, timeout: float = 5.0) -> None:
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


def drive_ml_worker_until_idle(*, timeout: float = 30.0) -> None:
    from ml_service.jobs.worker import process_next_job

    deadline = time.monotonic() + timeout
    idle_rounds = 0
    while time.monotonic() < deadline:
        if process_next_job():
            idle_rounds = 0
            continue
        idle_rounds += 1
        if idle_rounds >= 3:
            return
        time.sleep(0.02)
    raise AssertionError("ML worker did not become idle before timeout")


def wait_for_ml_jobs(
    *,
    product_job_id: uuid.UUID,
    count: int,
    timeout: float = 15.0,
) -> list[MLJob]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = list_ml_jobs(product_job_id=product_job_id)
        if len(jobs) >= count:
            return jobs
        time.sleep(0.05)
    raise AssertionError(
        f"expected {count} ml job(s) for product job {product_job_id}, "
        f"got {len(list_ml_jobs(product_job_id=product_job_id))}"
    )


def list_ml_jobs(*, product_job_id: uuid.UUID | None = None) -> list[MLJob]:
    from sqlalchemy import select

    from ml_service.infrastructure.orm_models import MLJob

    with SyncSessionLocal() as session:
        query = select(MLJob).order_by(MLJob.created_at)
        if product_job_id is not None:
            query = query.where(MLJob.product_job_id == product_job_id)
        return list(session.execute(query).scalars().all())
