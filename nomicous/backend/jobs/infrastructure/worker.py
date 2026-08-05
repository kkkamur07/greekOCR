"""Background job worker - polls Postgres and runs claimed jobs.

Multiple Uvicorn workers are safe: claim uses ``FOR UPDATE SKIP LOCKED`` so only
one process runs a given job. Parallelism here means *different* jobs at once,
not the same job twice.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from inference.contracts.jobs import JobSubmitRequest, JobSubmitResponse

from backend.core.settings.job import JobSettings, get_job_settings
from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
)
from backend.jobs.application.inference_dispatcher import build_inference_submit_request
from backend.jobs.infrastructure.handlers import TestJobHandlerError, run_test_handler
from backend.jobs.infrastructure.job_repository import (
    claim_next_pending_job,
    clear_stale_callback_claims,
    fail_stale_waiting_jobs,
    mark_job_done,
    mark_job_failed,
    mark_job_waiting,
    reclaim_stale_running_jobs,
    seconds_until_next_stale_waiting_job,
)
from backend.jobs.infrastructure.orm_models import Job, JobType
from backend.ml.infrastructure.ml_client import InferenceClient

if TYPE_CHECKING:
    from asyncio import Event

logger = logging.getLogger(__name__)

_inference_client: InferenceClient | None = None


def _get_inference_client() -> InferenceClient:
    global _inference_client
    if _inference_client is None:
        _inference_client = InferenceClient()
    return _inference_client


def _submit_inference_job(request: JobSubmitRequest) -> JobSubmitResponse:
    return _get_inference_client().submit_job(request)


def _public_job_error(exc: BaseException, *, fallback: str = "Job failed") -> str:
    """Return allowlisted job output; exception text stays server-side only."""
    if isinstance(exc, TranscribeJobHandlerError):
        return "Inference request could not be prepared"
    if isinstance(exc, TestJobHandlerError):
        return "Test job failed"
    return fallback


def execute_claimed_job(job: Job) -> None:
    """Run handler for a job already in ``running`` status."""
    # Terminal writes below are scoped to the claim this worker still holds, so a
    # job reclaimed mid-run cannot be finished twice. The inference callback path
    # completes ``waiting`` jobs from another process and does not go through here.
    owner = job.claimed_by

    if job.type in (JobType.segment, JobType.transcribe):
        try:
            request = build_inference_submit_request(job)
            submit_response = _submit_inference_job(request)
            mark_job_waiting(
                job.id,
                inference_job_id=submit_response.inference_job_id,
            )
        except TranscribeJobHandlerError as exc:
            logger.warning("Transcribe job %s failed: %s", job.id, exc)
            mark_job_failed(job.id, _public_job_error(exc), claimed_by=owner)
        except Exception as exc:
            logger.exception("%s job %s failed", job.type.value.title(), job.id, exc_info=exc)
            mark_job_failed(job.id, _public_job_error(exc), claimed_by=owner)
        return

    if (job.payload or {}).get("test"):
        try:
            result = run_test_handler(job)
        except TestJobHandlerError as exc:
            logger.warning("Test job %s failed: %s", job.id, exc)
            mark_job_failed(job.id, _public_job_error(exc), claimed_by=owner)
            return
        except Exception as exc:
            logger.exception("Test job %s failed", job.id, exc_info=exc)
            mark_job_failed(job.id, _public_job_error(exc), claimed_by=owner)
            return
        mark_job_done(job.id, result, claimed_by=owner)
        return

    logger.error("No handler for job %s type=%s", job.id, job.type.value)
    mark_job_failed(job.id, "Job type is not supported", claimed_by=owner)


def process_one_job() -> bool:
    """Claim and execute at most one job. Returns True if work was done."""
    settings = get_job_settings()
    reclaimed = reclaim_stale_running_jobs(
        running_timeout_seconds=settings.job_worker_running_timeout_seconds
    )
    if reclaimed:
        logger.warning("reclaimed %s stale platform job(s)", reclaimed)
    # Sweep waiting before releasing claims: releasing touches the row and would
    # otherwise push the waiting deadline out by another timeout window.
    timed_out = fail_stale_waiting_jobs(
        waiting_timeout_seconds=settings.job_worker_waiting_timeout_seconds
    )
    if timed_out:
        logger.warning("failed %s platform job(s) waiting past the inference timeout", timed_out)
    released = clear_stale_callback_claims(
        claim_timeout_seconds=settings.job_worker_callback_claim_timeout_seconds
    )
    if released:
        logger.warning("released %s stale inference callback claim(s)", released)
    job = claim_next_pending_job(test_only=settings.job_worker_claim_test_only)
    if job is None:
        return False
    execute_claimed_job(job)
    return True


async def _idle_wait_seconds(settings: JobSettings, idle_interval: float) -> float:
    """Cap the idle backoff so it can never sleep past a waiting job's deadline."""
    waiting_deadline = await asyncio.to_thread(
        seconds_until_next_stale_waiting_job,
        waiting_timeout_seconds=settings.job_worker_waiting_timeout_seconds,
    )
    if waiting_deadline is None:
        return idle_interval
    # Floor at the poll interval so an already-expired deadline cannot spin the loop.
    return max(min(idle_interval, waiting_deadline), settings.job_poll_interval_seconds)


async def worker_loop(stop_event: Event) -> None:
    """Poll for pending jobs until ``stop_event`` is set; backoff when idle."""
    settings = get_job_settings()
    min_interval = settings.job_poll_interval_seconds
    idle_interval = min_interval
    max_interval = settings.job_poll_max_interval_seconds

    while not stop_event.is_set():
        try:
            worked = await asyncio.to_thread(process_one_job)
        except Exception:
            logger.exception("job worker tick failed")
            worked = False
        if worked:
            idle_interval = min_interval
        else:
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=await _idle_wait_seconds(settings, idle_interval),
                )
            except TimeoutError:
                idle_interval = min(idle_interval * 2, max_interval)
