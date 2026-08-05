"""Job persistence - async enqueue/read; sync claim with SKIP LOCKED."""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from datetime import UTC, datetime, timedelta

import infrastructure.models  # noqa: F401 - register all ORM mappers
from infrastructure.db import sync_system_session
from sqlalchemy import delete, func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import PageCursor
from backend.core.exceptions import ConflictError
from backend.document.infrastructure.orm_models import Document
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType

_TERMINAL_STATUSES = (JobStatus.done, JobStatus.failed, JobStatus.cancelled)
_NON_TERMINAL_STATUSES = (JobStatus.pending, JobStatus.running, JobStatus.waiting)

WAITING_TIMEOUT_ERROR = "Inference timed out with no response"


def waiting_timeout_error(waiting_timeout_seconds: float) -> str:
    """Reason recorded when the inference service never called back.

    Allowlisted static text, same rule as ``worker._public_job_error``: no
    exception detail reaches the client. Naming the deadline is what separates
    "inference went silent for 240s" from every other failure, so the UI can say
    something honest instead of a generic error. The ``WAITING_TIMEOUT_ERROR``
    prefix stays stable so callers can still recognise the timeout.
    """
    return f"{WAITING_TIMEOUT_ERROR} after {int(waiting_timeout_seconds)}s"


_WORKER_IDENTITY = f"{socket.gethostname()}:{os.getpid()}"


def worker_identity() -> str:
    """Stable per-process worker id recorded on claim (host:pid)."""
    return _WORKER_IDENTITY


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

    async def delete_terminal_jobs_for_project(self, project_id: uuid.UUID) -> int:
        """Delete finished jobs of a project; pending/running/waiting rows are kept."""
        terminal_ids = (
            select(Job.id)
            .join(Document, Job.document_id == Document.id)
            .where(Document.project_id == project_id)
            .where(Job.status.in_(_TERMINAL_STATUSES))
            .scalar_subquery()
        )
        result = await self._session.execute(delete(Job).where(Job.id.in_(terminal_ids)))
        await self._session.commit()
        return result.rowcount or 0


# Job types an inference agent claims for itself over HTTP. The platform worker
# leaves them alone: with the second queue gone (ADR 0003) it has no way to run
# them, and claiming one would only fail a job that an agent could have run.
AGENT_CLAIMED_JOB_TYPES = (JobType.segment, JobType.transcribe)


def _pending_job_query(*, test_only: bool | None = None):
    query = (
        select(Job)
        .where(Job.status == JobStatus.pending)
        .where(Job.type.not_in(AGENT_CLAIMED_JOB_TYPES))
        .order_by(Job.created_at, Job.id)
    )
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
        job.claimed_by = worker_identity()
        job.heartbeat_at = now
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
                claimed_by=None,
                heartbeat_at=None,
                updated_at=now,
            )
        )
        session.commit()
        return result.rowcount or 0


def fail_stale_waiting_jobs(*, waiting_timeout_seconds: float) -> int:
    """Fail jobs stuck in ``waiting`` because the inference callback never arrived.

    Unlike a crashed *running* job there is nothing to retry: the dispatch
    already happened and the inference service went silent, so re-pending would
    duplicate work. Fail instead, with an error the user can act on.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=waiting_timeout_seconds)
    with sync_system_session() as session:
        stale_ids = list(
            session.execute(
                select(Job.id)
                .where(Job.status == JobStatus.waiting)
                .where(Job.updated_at <= stale_before)
                .with_for_update(skip_locked=True)
            ).scalars()
        )
        if not stale_ids:
            return 0
        result = session.execute(
            update(Job)
            .where(Job.id.in_(stale_ids))
            .where(Job.status == JobStatus.waiting)
            .values(
                status=JobStatus.failed,
                error=waiting_timeout_error(waiting_timeout_seconds),
                callback_claimed_at=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so SSE subscribers would keep
        # waiting on a job that just died. Notify each id after the commit.
        for job_id in stale_ids:
            notify_platform_job_status_changed(job_id, JobStatus.failed)
        return result.rowcount or 0


def clear_stale_callback_claims(*, claim_timeout_seconds: float) -> int:
    """Release abandoned inference callback claims on non-terminal jobs.

    A callback that crashes after claiming never clears ``callback_claimed_at``,
    which makes the job permanently uncancellable. Dropping the stale claim
    hands control back to the user.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=claim_timeout_seconds)
    with sync_system_session() as session:
        result = session.execute(
            update(Job)
            .where(Job.status.in_(_NON_TERMINAL_STATUSES))
            .where(Job.callback_claimed_at.is_not(None))
            .where(Job.callback_claimed_at <= stale_before)
            # Keep ``updated_at`` as-is: the column's ``onupdate`` would otherwise
            # bump it and push the waiting-timeout deadline back another window.
            .values(callback_claimed_at=None, updated_at=Job.updated_at)
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


def seconds_until_next_stale_waiting_job(*, waiting_timeout_seconds: float) -> float | None:
    """Return seconds until the oldest waiting job is eligible for the timeout sweep."""
    with sync_system_session() as session:
        oldest_updated_at = session.execute(
            select(func.min(Job.updated_at)).where(Job.status == JobStatus.waiting)
        ).scalar_one_or_none()
    if oldest_updated_at is None:
        return None

    now = datetime.now(UTC)
    fail_at = oldest_updated_at + timedelta(seconds=waiting_timeout_seconds)
    return max((fail_at - now).total_seconds(), 0.0)


def mark_job_waiting(
    job_id: uuid.UUID,
    *,
    inference_job_id: uuid.UUID | None = None,
    payload_patch: dict | None = None,
) -> None:
    """Move a non-terminal job to ``waiting``. No-op if already terminal.

    Uses ``FOR UPDATE`` so a concurrent cancel cannot be overwritten by the
    status write after a stale unlocked read.
    """
    now = datetime.now(UTC)
    with sync_system_session() as session:
        job = session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        ).scalar_one_or_none()
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


def _owned_by(statement, claimed_by: str | None):
    """Restrict a worker's terminal write to the claim it still owns.

    ``reclaim_stale_running_jobs`` clears ``claimed_by`` and re-pends the job, so
    a zombie worker that lost its lease no longer matches: either the column is
    NULL (not yet re-claimed) or it holds the new owner's id. Status alone is not
    enough, because the reclaimed job is legitimately non-terminal again.
    """
    return statement.where(Job.claimed_by.is_not_distinct_from(claimed_by))


def mark_job_failed(job_id: uuid.UUID, error: str, *, claimed_by: str | None) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        update_result = session.execute(
            _owned_by(
                update(Job)
                .where(Job.id == job_id)
                .where(Job.status.notin_((JobStatus.cancelled, JobStatus.done))),
                claimed_by,
            ).values(
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


def mark_job_done(job_id: uuid.UUID, result: dict | None = None, *, claimed_by: str | None) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        update_result = session.execute(
            _owned_by(
                update(Job)
                .where(Job.id == job_id)
                .where(Job.status.notin_((JobStatus.cancelled, JobStatus.failed))),
                claimed_by,
            ).values(
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


_CANCELABLE = _NON_TERMINAL_STATUSES


def _apply_cancellation(job: Job, now: datetime) -> None:
    if job.status not in _CANCELABLE:
        raise ConflictError(f"job {job.id} cannot be cancelled from status {job.status.value}")
    if job.callback_claimed_at is not None:
        raise ConflictError(
            f"job {job.id} cannot be cancelled while an inference callback is applied"
        )
    job.status = JobStatus.cancelled
    job.error = None
    job.result = None
    job.callback_claimed_at = None
    job.completed_at = now
    job.updated_at = now


async def cancel_job_async(session: AsyncSession, job_id: uuid.UUID) -> Job | None:
    """Atomically cancel a pending/running/waiting job under ``FOR UPDATE``.

    Raises ConflictError when the job is already terminal or a callback has been
    claimed (merge may already be applying document changes).
    """
    now = datetime.now(UTC)
    job = (
        await session.execute(select(Job).where(Job.id == job_id).with_for_update())
    ).scalar_one_or_none()
    if job is None:
        return None
    _apply_cancellation(job, now)
    await session.commit()
    await session.refresh(job)
    # pg_notify uses a sync DB session - keep it off the event loop.
    await asyncio.to_thread(notify_platform_job_status_changed, job.id, JobStatus.cancelled)
    return job
