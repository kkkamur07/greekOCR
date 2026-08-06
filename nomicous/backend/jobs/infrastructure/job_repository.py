"""Job persistence - async enqueue/read; sync claim with SKIP LOCKED."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from datetime import UTC, datetime, timedelta

import infrastructure.models  # noqa: F401 - register all ORM mappers
from infrastructure.db import sync_system_session
from sqlalchemy import delete, func, or_, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import PageCursor
from backend.core.exceptions import ConflictError
from backend.document.infrastructure.orm_models import Document
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = (JobStatus.done, JobStatus.failed, JobStatus.cancelled)
_NON_TERMINAL_STATUSES = (JobStatus.pending, JobStatus.running, JobStatus.waiting)

WAITING_TIMEOUT_ERROR = "Inference timed out with no response"

POISON_PAGE_ERROR = "This page could not be completed by any inference agent"

MAX_CLAIM_ATTEMPTS = 5
"""How many abandoned claims a page survives before it is failed.

Both stale sweeps return an abandoned job to ``pending`` rather than failing it,
and that is the right default: a closed laptop lid is not a failed job. But a
page that reliably kills whatever runs it - a corrupt scan, an image the model
crashes on - is abandoned for the same reason every time, so re-pending it is an
infinite loop that never tells the researcher anything and burns one agent slot
per lap. ``jobs.claim_attempts`` counts the laps and this is where they stop.

A constant rather than an environment dial, unlike the two timeouts next to it.
Those are operational: how long to wait before believing an agent is gone
depends on the deployment. This is not - "a page that has already killed five
agents will not be run by a sixth" is a property of the queue, and an operator
who raises it is choosing a longer infinite loop.
"""


def poison_page_error(max_claim_attempts: int) -> str:
    """Reason recorded when a page exhausted its claim budget.

    Allowlisted static text, same rule as ``waiting_timeout_error``: nothing the
    page itself produced reaches the client, because the whole reason this fires
    is that no agent ever got far enough to report anything. Naming the count is
    what separates "five agents took this page and none came back" from a
    generic failure, so the UI can tell a researcher the page is the problem
    rather than the platform.
    """
    return f"{POISON_PAGE_ERROR} (abandoned {max_claim_attempts} times)"


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

# The prefix ``jobs.claimed_by`` carries while an **inference agent** holds a page,
# written by ``job_claim_service.agent_claim_owner``. It lives down here, next to
# the sweeps that read it, because "which timeout governs this row" is answered
# from the column alone and the sweeps must not import the application layer.
AGENT_CLAIM_PREFIX = "agent:"


def _held_by_agent():
    """SQL for "an inference agent holds this page"."""
    return Job.claimed_by.startswith(AGENT_CLAIM_PREFIX)


def _not_held_by_agent():
    """SQL for the complement, NULL included.

    ``NOT LIKE`` alone evaluates to NULL for an unclaimed row and would silently
    exempt every job the platform itself dispatched - which is the whole
    population the waiting timeout exists for.
    """
    return or_(Job.claimed_by.is_(None), ~Job.claimed_by.startswith(AGENT_CLAIM_PREFIX))


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


def _split_by_claim_budget(
    rows: list, max_claim_attempts: int
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """Split ``(id, claim_attempts)`` rows into (retryable, exhausted).

    The comparison is against ``attempts + 1`` because the caller is about to
    record *this* abandonment: a row already at the ceiling would otherwise be
    re-pended one last time and only fail on the lap after.
    """
    retryable: list[uuid.UUID] = []
    exhausted: list[uuid.UUID] = []
    for job_id, attempts in rows:
        target = exhausted if (attempts or 0) + 1 >= max_claim_attempts else retryable
        target.append(job_id)
    return retryable, exhausted


def reclaim_stale_running_jobs(
    *, running_timeout_seconds: float, max_claim_attempts: int = MAX_CLAIM_ATTEMPTS
) -> int:
    """Take back a crashed worker's claim once its running lease expires.

    Almost always that means re-pending, so another worker runs the job. A job
    whose claim has been abandoned ``max_claim_attempts`` times is failed
    instead: see ``MAX_CLAIM_ATTEMPTS`` for why re-pending forever is not the
    kinder option.

    ``FOR UPDATE SKIP LOCKED`` on the select, the same primitive the other two
    sweeps use. It is not only concurrency control here - the split below needs
    each row's ``claim_attempts``, and reading it unlocked would let a second
    sweeper decide from the same stale count.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=running_timeout_seconds)
    with sync_system_session() as session:
        rows = list(
            session.execute(
                select(Job.id, Job.claim_attempts)
                .where(Job.status == JobStatus.running)
                .where(Job.started_at <= stale_before)
                .with_for_update(skip_locked=True)
            )
        )
        if not rows:
            return 0
        retryable, exhausted = _split_by_claim_budget(rows, max_claim_attempts)
        moved = 0
        if retryable:
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(retryable))
                    .where(Job.status == JobStatus.running)
                    .values(
                        status=JobStatus.pending,
                        started_at=None,
                        claimed_by=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        if exhausted:
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(exhausted))
                    .where(Job.status == JobStatus.running)
                    .values(
                        status=JobStatus.failed,
                        error=poison_page_error(max_claim_attempts),
                        claimed_by=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        callback_claimed_at=None,
                        completed_at=now,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so an SSE subscriber would sit on
        # a job that just died. The re-pended rows are deliberately not announced:
        # this sweep has never announced them, and pending is not a state anything
        # renders differently from the running it replaces.
        for job_id in exhausted:
            logger.warning("failing job %s: claim abandoned %s times", job_id, max_claim_attempts)
            notify_platform_job_status_changed(job_id, JobStatus.failed)
        return moved


def fail_stale_waiting_jobs(*, waiting_timeout_seconds: float) -> int:
    """Fail jobs stuck in ``waiting`` because the inference callback never arrived.

    Unlike a crashed *running* job there is nothing to retry: the dispatch
    already happened and the inference service went silent, so re-pending would
    duplicate work. Fail instead, with an error the user can act on.

    **Agent-held pages are deliberately not in this population.** ADR 0005 makes a
    claimed page ``waiting`` so ``JobCallbackService`` needs no change, which put
    a researcher's laptop under this timeout by accident. The reasoning above does
    not hold for one: nothing was dispatched anywhere, the page never left the
    queue, and a closed lid is not a silent inference service. Those rows are
    governed by ``release_expired_device_leases``, which re-pends instead of
    failing - see ``AGENT_CLAIM_PREFIX``.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=waiting_timeout_seconds)
    with sync_system_session() as session:
        stale_ids = list(
            session.execute(
                select(Job.id)
                .where(Job.status == JobStatus.waiting)
                .where(_not_held_by_agent())
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
            .where(_not_held_by_agent())
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


def release_expired_device_leases(
    *, lease_seconds: float, max_claim_attempts: int = MAX_CLAIM_ATTEMPTS
) -> int:
    """Return pages whose **lease** expired to the queue, so another agent can take them.

    A crash, a killed process, or a closed laptop lid leaves a page held by an
    agent that will never report on it. There is no heartbeat to notice - work is
    seconds-to-minutes, so the lease covers it with margin (ADR 0002) - and there
    is no release endpoint, because a process that was killed cannot call one.
    This sweep is the whole recovery mechanism.

    **It re-pends; it does not fail.** That is the difference from
    ``fail_stale_waiting_jobs``, and it is the point of the lease. A researcher
    who closes their lid mid-page should have that page picked up by the next
    agent, not see it permanently failed and have to resubmit. Nothing was
    dispatched to a third party, so there is no duplicate work to fear: the page
    is still exactly where it started, in the platform's own queue.

    **Except after ``max_claim_attempts`` laps.** Re-pending is right when the
    agent is why the page came back; it is a loop when the *page* is. A scan the
    model crashes on takes down every agent that claims it, so with no counter it
    cycled ``pending -> waiting -> pending`` forever: never terminal, so never
    reported to the researcher, and holding one claim slot on every lap. The
    counter lives on the row (``jobs.claim_attempts``) so this bulk statement can
    read and bump it without a per-row round trip. See ``MAX_CLAIM_ATTEMPTS``.

    The claim is cleared with it - ``claimed_by``, ``inference_job_id``,
    ``started_at``, ``heartbeat_at`` - so that *any* agent may take the page next,
    and so a woken zombie cannot report on it: the callback route matches
    ``claimed_by``, and ``JobCallbackService`` matches ``inference_job_id``.
    Neither matches once this has run.

    Concurrency is the same primitive the other sweeps use. ``FOR UPDATE SKIP
    LOCKED`` in the same transaction as the update means a second sweeper skips
    rows already being released rather than releasing them twice, and the
    ``status``/``claimed_by`` predicates are repeated on the update so a row that
    changed hands between the two statements is left alone.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=lease_seconds)
    with sync_system_session() as session:
        rows = list(
            session.execute(
                select(Job.id, Job.claim_attempts)
                .where(Job.status == JobStatus.waiting)
                .where(_held_by_agent())
                .where(Job.updated_at <= stale_before)
                .with_for_update(skip_locked=True)
            )
        )
        if not rows:
            return 0
        retryable, exhausted = _split_by_claim_budget(rows, max_claim_attempts)
        moved = 0
        if retryable:
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(retryable))
                    .where(Job.status == JobStatus.waiting)
                    .where(_held_by_agent())
                    .values(
                        status=JobStatus.pending,
                        claimed_by=None,
                        inference_job_id=None,
                        started_at=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        callback_claimed_at=None,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        if exhausted:
            # The claim is cleared here too, for the same reason it is cleared on a
            # re-pend: a woken zombie must not be able to report on the page. A
            # terminal status alone would not stop it, because the callback path
            # matches ``inference_job_id``, not ``status``.
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(exhausted))
                    .where(Job.status == JobStatus.waiting)
                    .where(_held_by_agent())
                    .values(
                        status=JobStatus.failed,
                        error=poison_page_error(max_claim_attempts),
                        claimed_by=None,
                        inference_job_id=None,
                        started_at=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        callback_claimed_at=None,
                        completed_at=now,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so a browser watching the job
        # would sit on "waiting" until something else touched the row.
        for job_id in retryable:
            notify_platform_job_status_changed(job_id, JobStatus.pending)
        for job_id in exhausted:
            logger.warning("failing job %s: claim abandoned %s times", job_id, max_claim_attempts)
            notify_platform_job_status_changed(job_id, JobStatus.failed)
        return moved


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
                # A page that finished is not a page anything struggled with. The
                # counter is reset rather than left standing so a job that took
                # four tries and then succeeded does not carry a one-lap budget
                # into whatever re-pends it next.
                claim_attempts=0,
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
