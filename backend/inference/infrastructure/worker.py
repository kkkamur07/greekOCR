"""Background job worker — polls Postgres and runs claimed jobs.

Multiple Uvicorn workers are safe: claim uses ``FOR UPDATE SKIP LOCKED`` so only
one process runs a given job.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

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

POLL_INTERVAL_SECONDS = 0.05


def execute_claimed_job(job: Job) -> None:
    """Run handler for a job already in ``running`` status."""
    if (job.payload or {}).get("test"):
        try:
            result = run_test_handler(job)
        except TestJobHandlerError as exc:
            mark_job_failed(job.id, str(exc))
            return
        except Exception as exc:
            mark_job_failed(job.id, str(exc))
            return
        mark_job_done(job.id, result)
        return
    mark_job_failed(job.id, f"no handler for job type {job.type.value}")


def process_one_job() -> bool:
    """Claim and execute at most one job. Returns True if work was done."""
    job = claim_next_pending_job(test_only=True)
    if job is None:
        return False
    execute_claimed_job(job)
    return True


async def worker_loop(stop_event: Event) -> None:
    """Poll for pending jobs until ``stop_event`` is set."""
    while not stop_event.is_set():
        try:
            worked = await asyncio.to_thread(process_one_job)
        except Exception:
            logger.exception("job worker tick failed")
            worked = False
        if not worked:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL_SECONDS)
            except TimeoutError:
                pass
