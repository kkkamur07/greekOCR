"""Health check routes."""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.schemas.health import HealthResponse
from backend.core.settings.job import get_job_settings
from backend.jobs.infrastructure.job_repository import JobRepository
from infrastructure.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


async def _oldest_pending_job_seconds(db: AsyncSession) -> float | None:
    """Read the queue's head age and warn when it is past the stall threshold.

    The WARNING is the part that actually alerts. Nobody reads a probe's response
    body, so a number alone would be as silent as the queue it describes; a
    WARNING line lands in log-based alerting with no extra infrastructure. It is
    emitted on every probe past the threshold rather than once per crossing,
    deliberately: alerting wants a rate, and a warning that stops arriving must
    mean the queue drained, not that a process-local latch is still set.

    A failure to read the queue is logged with its traceback and reported as
    ``None`` rather than raised. ``SELECT 1`` has already succeeded by the time
    this runs, so the probe's own subject - can this process reach Postgres - is
    answered; failing the request here would take the API out of rotation over a
    supplementary number.
    """
    try:
        age_seconds = await JobRepository(db).seconds_since_oldest_pending_job()
    except SQLAlchemyError:
        logger.exception("health check could not read the pending job queue")
        return None
    threshold = get_job_settings().job_queue_stall_warning_seconds
    if age_seconds is not None and age_seconds >= threshold:
        logger.warning(
            "oldest pending job has waited %.0fs (threshold %.0fs): "
            "no platform worker or inference agent is claiming the queue",
            age_seconds,
            threshold,
        )
    return age_seconds


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unreachable"}},
)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse | JSONResponse:
    """Liveness/readiness (``SELECT 1``) plus the age of the oldest pending job.

    Dev-user seeding lives in app lifespan.

    **The queue age never changes the status code.** A pending job piles up
    because a *different* host stopped claiming - the platform worker
    (``JOB_WORKER_ENABLED``, which the API deployment sets false) or a
    researcher's inference agent. Returning 503 for that would pull a perfectly
    healthy API out of rotation, or restart it, and neither brings the missing
    consumer back. The status code answers "can this process serve requests";
    the queue age is reported next to it, and the WARNING in
    ``_oldest_pending_job_seconds`` is what pages someone.

    It rides on ``/health`` rather than a separate ``/health/jobs`` because the
    thing that has to happen on a schedule is the threshold check, and ``/health``
    is the only route infrastructure already polls on a schedule. A route nobody
    is configured to call would put the queue's one alarm behind another manual
    deployment step - the same class of gap that leaves the queue unclaimed in
    the first place. The cost is one ``min(created_at)`` against the
    ``ix_jobs_claim_pending`` partial index per probe.
    """
    try:
        await db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        body = HealthResponse(status="degraded", database="error")
        return JSONResponse(status_code=503, content=body.model_dump())

    return HealthResponse(
        status="ok",
        database="ok",
        oldest_pending_job_seconds=await _oldest_pending_job_seconds(db),
    )
