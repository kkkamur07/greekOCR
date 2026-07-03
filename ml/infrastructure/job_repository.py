"""ML job queue persistence — sync claim with FOR UPDATE SKIP LOCKED."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from ml.contracts.common import MLJobStatus
from ml.contracts.jobs import JobSubmitRequest
from ml.infrastructure.db import SessionLocal
from ml.infrastructure.orm_models import MLJob


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
                .order_by(MLJob.created_at)
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
