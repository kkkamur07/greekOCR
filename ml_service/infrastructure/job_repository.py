"""ML job queue persistence — sync claim with FOR UPDATE SKIP LOCKED."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text, update

from ml_service.contracts.common import MLJobStatus
from ml_service.contracts.jobs import JobSubmitRequest
from ml_service.infrastructure.db import SessionLocal
from ml_service.infrastructure.orm_models import MLJob
from ml_service.infrastructure.settings import get_ml_settings


def create_job(request: JobSubmitRequest) -> MLJob:
    job = MLJob(
        product_job_id=request.product_job_id,
        task=request.task,
        registry_model_id=request.registry_model_id,
        registry_tag=request.registry_tag,
        status=MLJobStatus.pending,
        image_bytes=request.image_bytes,
        params=request.params,
    )
    with SessionLocal() as session:
        session.add(job)
        session.flush()
        session.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {
                "channel": get_ml_settings().worker_notify_channel,
                "payload": str(job.id),
            },
        )
        session.commit()
        session.refresh(job)
        return job


def get_job_by_id(job_id: uuid.UUID) -> MLJob | None:
    with SessionLocal() as session:
        return session.get(MLJob, job_id)


def claim_next_pending_job() -> MLJob | None:
    """Claim one pending ML job using FOR UPDATE SKIP LOCKED."""
    with SessionLocal() as session:
        job = (
            session.execute(
                select(MLJob)
                .where(MLJob.status == MLJobStatus.pending)
                .order_by(MLJob.created_at, MLJob.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            .scalar_one_or_none()
        )
        if job is None:
            return None
        now = datetime.now(UTC)
        job.status = MLJobStatus.running
        job.started_at = now
        job.updated_at = now
        session.commit()
        session.refresh(job)
        return job


def reclaim_stale_running_jobs(*, running_timeout_seconds: float) -> int:
    """Move crashed-worker jobs back to pending after their running lease expires."""
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=running_timeout_seconds)
    with SessionLocal() as session:
        result = session.execute(
            update(MLJob)
            .where(MLJob.status == MLJobStatus.running)
            .where(MLJob.started_at <= stale_before)
            .values(
                status=MLJobStatus.pending,
                started_at=None,
                updated_at=now,
            )
        )
        session.commit()
        return result.rowcount or 0


def seconds_until_next_stale_running_job(*, running_timeout_seconds: float) -> float | None:
    """Return seconds until the oldest running job is eligible for reclaim."""
    with SessionLocal() as session:
        oldest_started_at = session.execute(
            select(func.min(MLJob.started_at)).where(MLJob.status == MLJobStatus.running)
        ).scalar_one_or_none()
    if oldest_started_at is None:
        return None

    now = datetime.now(UTC)
    reclaim_at = oldest_started_at + timedelta(seconds=running_timeout_seconds)
    return max((reclaim_at - now).total_seconds(), 0.0)


def mark_job_done(job_id: uuid.UUID, output: dict[str, Any]) -> None:
    now = datetime.now(UTC)
    with SessionLocal() as session:
        session.execute(
            update(MLJob)
            .where(MLJob.id == job_id)
            .values(
                status=MLJobStatus.done,
                output=output,
                error=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()


def mark_job_failed(job_id: uuid.UUID, error: str) -> None:
    now = datetime.now(UTC)
    with SessionLocal() as session:
        session.execute(
            update(MLJob)
            .where(MLJob.id == job_id)
            .values(
                status=MLJobStatus.failed,
                error=error,
                output=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
