"""Inference job queue persistence - sync claim with FOR UPDATE SKIP LOCKED."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text, update

from inference.contracts.common import InferenceJobStatus
from inference.contracts.jobs import JobSubmitRequest
from inference.infrastructure.db import SessionLocal
from inference.infrastructure.orm_models import InferenceJob
from inference.infrastructure.settings import get_inference_settings

_QUEUE_ADMISSION_LOCK_KEY = 8_402_761


class QueueSaturatedError(RuntimeError):
    """Raised when the bounded inference queue cannot accept more work."""


def create_job(request: JobSubmitRequest) -> InferenceJob:
    job = InferenceJob(
        product_job_id=request.product_job_id,
        task=request.task,
        registry_model_id=request.registry_model_id,
        registry_tag=request.registry_tag,
        status=InferenceJobStatus.pending,
        image_bytes=request.image_bytes,
        params=request.params,
    )
    with SessionLocal() as session:
        # Serialize the count-and-insert admission decision across API processes.
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": _QUEUE_ADMISSION_LOCK_KEY},
        )
        active_jobs = session.execute(
            select(func.count())
            .select_from(InferenceJob)
            .where(
                InferenceJob.status.in_([InferenceJobStatus.pending, InferenceJobStatus.running])
            )
        ).scalar_one()
        if active_jobs >= get_inference_settings().inference_max_pending_jobs:
            raise QueueSaturatedError("inference queue is saturated")
        session.add(job)
        session.flush()
        session.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {
                "channel": get_inference_settings().worker_notify_channel,
                "payload": str(job.id),
            },
        )
        session.commit()
        session.refresh(job)
        return job


def get_job_by_id(job_id: uuid.UUID) -> InferenceJob | None:
    with SessionLocal() as session:
        return session.get(InferenceJob, job_id)


def claim_next_pending_job() -> InferenceJob | None:
    """Claim one pending inference job using FOR UPDATE SKIP LOCKED."""
    with SessionLocal() as session:
        job = session.execute(
            select(InferenceJob)
            .where(InferenceJob.status == InferenceJobStatus.pending)
            .order_by(InferenceJob.created_at, InferenceJob.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        ).scalar_one_or_none()
        if job is None:
            return None
        now = datetime.now(UTC)
        job.status = InferenceJobStatus.running
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
            update(InferenceJob)
            .where(InferenceJob.status == InferenceJobStatus.running)
            .where(InferenceJob.started_at <= stale_before)
            .values(
                status=InferenceJobStatus.pending,
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
            select(func.min(InferenceJob.started_at)).where(
                InferenceJob.status == InferenceJobStatus.running
            )
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
            update(InferenceJob)
            .where(InferenceJob.id == job_id)
            .values(
                status=InferenceJobStatus.done,
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
            update(InferenceJob)
            .where(InferenceJob.id == job_id)
            .values(
                status=InferenceJobStatus.failed,
                error=error,
                output=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
