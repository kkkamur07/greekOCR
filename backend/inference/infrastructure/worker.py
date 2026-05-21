"""Background job worker — polls Postgres and runs claimed jobs.

Multiple Uvicorn workers are safe: claim uses ``FOR UPDATE SKIP LOCKED`` so only
one process runs a given job. Parallelism here means *different* jobs at once,
not the same job twice.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from backend.core.settings.job import get_job_settings
from backend.inference.infrastructure.handlers import TestJobHandlerError, run_test_handler
from backend.inference.infrastructure.job_repository import (
    claim_next_pending_job,
    mark_job_done,
    mark_job_failed,
)
from backend.inference.infrastructure.orm_models import Job

if TYPE_CHECKING:
    from asyncio import Event

logger = logging.getLogger(__name__)


def _public_job_error(exc: BaseException, *, fallback: str = "Job failed") -> str:
    """User-safe message for Job.error; full detail goes to logs only."""
    if isinstance(exc, TestJobHandlerError):
        return str(exc)
    return fallback


def execute_claimed_job(job: Job) -> None:
    """Run handler for a job already in ``running`` status."""
    if (job.payload or {}).get("test"):
        try:
            result = run_test_handler(job)
        except TestJobHandlerError as exc:
            logger.warning("Test job %s failed: %s", job.id, exc)
            mark_job_failed(job.id, _public_job_error(exc))
            return
        except Exception as exc:
            logger.exception("Test job %s failed", job.id, exc_info=exc)
            mark_job_failed(job.id, _public_job_error(exc))
            return
        mark_job_done(job.id, result)
        return
    logger.error("No handler for job %s type=%s", job.id, job.type.value)
    mark_job_failed(job.id, f"No handler for job type {job.type.value}")


def process_one_job() -> bool:
    """Claim and execute at most one job. Returns True if work was done."""
    settings = get_job_settings()
    job = claim_next_pending_job(test_only=settings.job_worker_claim_test_only)
    if job is None:
        return False
    execute_claimed_job(job)
    return True


async def worker_loop(stop_event: Event) -> None:
    """Poll for pending jobs until ``stop_event`` is set; backoff when idle."""
    settings = get_job_settings()
    idle_interval = settings.job_poll_interval_seconds
    max_interval = settings.job_poll_max_interval_seconds

    while not stop_event.is_set():
        try:
            worked = await asyncio.to_thread(process_one_job)
        except Exception:
            logger.exception("job worker tick failed")
            worked = False
        if worked:
            idle_interval = settings.job_poll_interval_seconds
        else:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=idle_interval)
            except TimeoutError:
                idle_interval = min(idle_interval * 2, max_interval)
