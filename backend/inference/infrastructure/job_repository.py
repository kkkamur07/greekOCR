"""Job persistence — async enqueue/read; sync claim with SKIP LOCKED."""

from __future__ import annotations

import infrastructure.models  # noqa: F401 — register all ORM mappers

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.inference.infrastructure.orm_models import Job, JobStatus, JobType
from infrastructure.db import SyncSessionLocal


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_test_job(self, handler: str) -> Job:
        job = Job(
            type=JobType.pipeline,
            status=JobStatus.pending,
            payload={"handler": handler, "test": True},
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> Job | None:
        result = await self._session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()


def _pending_job_query(*, test_only: bool | None = None):
    query = select(Job).where(Job.status == JobStatus.pending).order_by(Job.created_at)
    if test_only is True:
        query = query.where(Job.payload.contains({"test": True}))
    elif test_only is False:
        query = query.where(~Job.payload.contains({"test": True}))
    return query.with_for_update(skip_locked=True).limit(1)


def claim_next_pending_job(*, test_only: bool | None = None) -> Job | None:
    """Claim one pending job using FOR UPDATE SKIP LOCKED (sync session)."""
    with SyncSessionLocal() as session:
        job = session.execute(_pending_job_query(test_only=test_only)).scalar_one_or_none()
        if job is None:
            return None
        now = datetime.now(UTC)
        job.status = JobStatus.running
        job.started_at = now
        job.updated_at = now
        session.commit()
        session.refresh(job)
        return job


def mark_job_done(job_id: uuid.UUID, result: dict | None = None) -> None:
    now = datetime.now(UTC)
    with SyncSessionLocal() as session:
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status=JobStatus.done,
                result=result or {},
                error=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()


def mark_job_failed(job_id: uuid.UUID, error: str) -> None:
    now = datetime.now(UTC)
    with SyncSessionLocal() as session:
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status=JobStatus.failed,
                error=error,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
