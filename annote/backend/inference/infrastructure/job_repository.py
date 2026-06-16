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

    async def create_segment_job(
        self,
        *,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        document_part_id: uuid.UUID,
        model_id: uuid.UUID | None = None,
        binding_id: uuid.UUID | None = None,
    ) -> Job:
        job = Job(
            type=JobType.segment,
            status=JobStatus.pending,
            payload={"adapter": "kraken_stub"},
            user_id=user_id,
            document_id=document_id,
            document_part_id=document_part_id,
            model_id=model_id,
            binding_id=binding_id,
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


def lock_first_pending_job_for_test() -> tuple[object, Job | None]:
    """Begin a transaction and lock the oldest pending row (no SKIP LOCKED).

    Returns ``(session, job)`` — caller must ``rollback()`` or ``commit()`` to release.
    Used only to prove SKIP LOCKED in tests.
    """
    session = SyncSessionLocal()
    job = (
        session.execute(
            select(Job)
            .where(Job.status == JobStatus.pending)
            .order_by(Job.created_at)
            .with_for_update()
            .limit(1)
        )
        .scalar_one_or_none()
    )
    return session, job


def count_active_jobs(*, test_payload: bool | None = None) -> int:
    """Count pending or running jobs (optionally filter by payload test flag)."""
    from sqlalchemy import func

    with SyncSessionLocal() as session:
        query = select(func.count()).select_from(Job).where(
            Job.status.in_((JobStatus.pending, JobStatus.running))
        )
        if test_payload is True:
            query = query.where(Job.payload.contains({"test": True}))
        elif test_payload is False:
            query = query.where(~Job.payload.contains({"test": True}))
        return session.execute(query).scalar_one()


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
