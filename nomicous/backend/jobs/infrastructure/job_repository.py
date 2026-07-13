"""Job persistence — async enqueue/read; sync claim with SKIP LOCKED."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import infrastructure.models  # noqa: F401 — register all ORM mappers
from infrastructure.db import sync_system_session
from sqlalchemy import func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import PageCursor
from backend.document.infrastructure.orm_models import Document
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_test_job(self, handler: str, *, user_id: uuid.UUID | None = None) -> Job:
        job = Job(
            type=JobType.pipeline,
            status=JobStatus.pending,
            payload={"handler": handler, "test": True},
            user_id=user_id,
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def record_local_job(
        self,
        *,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        document_part_id: uuid.UUID,
        job_type: JobType,
        registry_model_id: str,
        registry_tag: str,
        result: dict,
    ) -> Job:
        """Record a browser-orchestrated local inference run for project job history."""
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        job = Job(
            type=job_type,
            status=JobStatus.done,
            user_id=user_id,
            document_id=document_id,
            document_part_id=document_part_id,
            payload={
                "execution": "local",
                "registry_model_id": registry_model_id,
                "registry_tag": registry_tag,
            },
            result=result,
            started_at=now,
            completed_at=now,
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> Job | None:
        result = await self._session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    async def list_for_project(
        self,
        project_id: uuid.UUID,
        *,
        limit: int = 50,
        cursor: PageCursor | None = None,
    ) -> list[Job]:
        query = (
            select(Job)
            .join(Document, Job.document_id == Document.id)
            .where(Document.project_id == project_id)
            .where(~Job.payload.contains({"test": True}))
            .order_by(Job.created_at.desc(), Job.id.desc())
        )
        if cursor is not None:
            query = query.where(tuple_(Job.created_at, Job.id) < (cursor.created_at, cursor.id))
        query = query.limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())


def _pending_job_query(*, test_only: bool | None = None):
    query = select(Job).where(Job.status == JobStatus.pending).order_by(Job.created_at, Job.id)
    if test_only is True:
        query = query.where(Job.payload.contains({"test": True}))
    elif test_only is False:
        query = query.where(~Job.payload.contains({"test": True}))
    return query.with_for_update(skip_locked=True).limit(1)


def claim_next_pending_job(*, test_only: bool | None = None) -> Job | None:
    """Claim one pending job using FOR UPDATE SKIP LOCKED (sync session)."""
    with sync_system_session() as session:
        job = session.execute(_pending_job_query(test_only=test_only)).scalar_one_or_none()
        if job is None:
            return None
        now = datetime.now(UTC)
        job.status = JobStatus.running
        job.started_at = now
        job.updated_at = now
        session.commit()
        session.refresh(job)
        notify_platform_job_status_changed(job.id, job.status)
        return job


def count_active_jobs(*, test_payload: bool | None = None) -> int:
    """Count pending, running, or waiting jobs (optionally filter by payload test flag)."""
    from sqlalchemy import func

    with sync_system_session() as session:
        query = (
            select(func.count())
            .select_from(Job)
            .where(Job.status.in_((JobStatus.pending, JobStatus.running, JobStatus.waiting)))
        )
        if test_payload is True:
            query = query.where(Job.payload.contains({"test": True}))
        elif test_payload is False:
            query = query.where(~Job.payload.contains({"test": True}))
        return session.execute(query).scalar_one()


def reclaim_stale_running_jobs(*, running_timeout_seconds: float) -> int:
    """Move crashed-worker jobs back to pending after their running lease expires."""
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=running_timeout_seconds)
    with sync_system_session() as session:
        result = session.execute(
            update(Job)
            .where(Job.status == JobStatus.running)
            .where(Job.started_at <= stale_before)
            .values(
                status=JobStatus.pending,
                started_at=None,
                updated_at=now,
            )
        )
        session.commit()
        return result.rowcount or 0


def seconds_until_next_stale_running_job(*, running_timeout_seconds: float) -> float | None:
    """Return seconds until the oldest running job is eligible for reclaim."""
    with sync_system_session() as session:
        oldest_started_at = session.execute(
            select(func.min(Job.started_at)).where(Job.status == JobStatus.running)
        ).scalar_one_or_none()
    if oldest_started_at is None:
        return None

    now = datetime.now(UTC)
    reclaim_at = oldest_started_at + timedelta(seconds=running_timeout_seconds)
    return max((reclaim_at - now).total_seconds(), 0.0)


def mark_job_waiting(
    job_id: uuid.UUID,
    *,
    inference_job_id: uuid.UUID | None = None,
    payload_patch: dict | None = None,
) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise ValueError(f"job {job_id} not found")
        if job.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
            return
        payload = dict(job.payload or {})
        if payload_patch:
            payload.update(payload_patch)
        job.payload = payload
        job.status = JobStatus.waiting
        job.callback_claimed_at = None
        if inference_job_id is not None:
            job.inference_job_id = inference_job_id
        job.updated_at = now
        session.commit()
        notify_platform_job_status_changed(job.id, job.status)


def mark_job_failed(job_id: uuid.UUID, error: str) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        update_result = session.execute(
            update(Job)
            .where(Job.id == job_id)
            .where(Job.status.notin_((JobStatus.cancelled, JobStatus.done)))
            .values(
                status=JobStatus.failed,
                error=error,
                callback_claimed_at=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
    if update_result.rowcount:
        notify_platform_job_status_changed(job_id, JobStatus.failed)


def mark_job_done(job_id: uuid.UUID, result: dict | None = None) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        update_result = session.execute(
            update(Job)
            .where(Job.id == job_id)
            .where(Job.status.notin_((JobStatus.cancelled, JobStatus.failed)))
            .values(
                status=JobStatus.done,
                result=result or {},
                error=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
    if update_result.rowcount:
        notify_platform_job_status_changed(job_id, JobStatus.done)


_CANCELABLE = (JobStatus.pending, JobStatus.running, JobStatus.waiting)


def cancel_job(job_id: uuid.UUID) -> Job | None:
    """Cancel a pending/running/waiting job and discard unapplied partials.

    Returns the job row. When the job was already terminal, returns it unchanged
    so callers can distinguish missing vs not-cancelable.
    """
    now = datetime.now(UTC)
    with sync_system_session() as session:
        job = session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        ).scalar_one_or_none()
        if job is None:
            return None
        if job.status not in _CANCELABLE:
            session.expunge(job)
            return job
        # Discard partials: clear result/callback claim so later inference
        # callbacks are ignored (_TERMINAL_STATUSES includes cancelled).
        job.status = JobStatus.cancelled
        job.error = None
        job.result = None
        job.callback_claimed_at = None
        job.completed_at = now
        job.updated_at = now
        session.commit()
        session.refresh(job)
        session.expunge(job)
        notify_platform_job_status_changed(job.id, JobStatus.cancelled)
        return job
