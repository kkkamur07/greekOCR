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
from backend.document.application.segment_merge_service import SegmentMergeService
from backend.document.infrastructure.media_store import MediaStore
from backend.document.infrastructure.orm_models import DocumentPart
from backend.ml.infrastructure.ml_client import MlServiceClient
from backend.jobs.infrastructure.handlers import (
    TestJobHandlerError,
    TranscribeJobHandlerError,
    run_test_handler,
    run_transcribe_handler,
)
from backend.jobs.infrastructure.job_repository import (
    claim_next_pending_job,
    mark_job_done,
    mark_job_failed,
)
from backend.jobs.infrastructure.orm_models import Job, JobType
from infrastructure.db import SyncSessionLocal

if TYPE_CHECKING:
    from asyncio import Event

logger = logging.getLogger(__name__)

# Registry/binding resolution is deferred; this slice only moves segment execution to ML HTTP.
_DEFAULT_SEGMENT_REGISTRY_MODEL = "kraken-blla"
_DEFAULT_SEGMENT_REGISTRY_TAG = "stable"
_ml_client: MlServiceClient | None = None


def _get_ml_client() -> MlServiceClient:
    global _ml_client
    if _ml_client is None:
        _ml_client = MlServiceClient()
    return _ml_client


def _public_job_error(exc: BaseException, *, fallback: str = "Job failed") -> str:
    """User-safe message for Job.error; full detail goes to logs only."""
    if isinstance(exc, (TestJobHandlerError, TranscribeJobHandlerError)):
        return str(exc)
    return fallback


def execute_claimed_job(job: Job) -> None:
    """Run handler for a job already in ``running`` status."""
    if job.type == JobType.transcribe:
        try:
            result = run_transcribe_handler(job)
        except TranscribeJobHandlerError as exc:
            logger.warning("Transcribe job %s failed: %s", job.id, exc)
            mark_job_failed(job.id, _public_job_error(exc))
            return
        except Exception as exc:
            logger.exception("Transcribe job %s failed", job.id, exc_info=exc)
            mark_job_failed(job.id, _public_job_error(exc))
            return
        mark_job_done(job.id, result)
        return

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
    if job.type == JobType.segment:
        try:
            with SyncSessionLocal() as session:
                if job.document_part_id is None:
                    raise ValueError("Segment job is missing its target document part")
                part = session.get(DocumentPart, job.document_part_id)
                if part is None:
                    raise ValueError("Document part not found")
                image_path = MediaStore().absolute_path(part.image_key)
                segment_output = _get_ml_client().run_segment(
                    registry_model_id=_DEFAULT_SEGMENT_REGISTRY_MODEL,
                    registry_tag=_DEFAULT_SEGMENT_REGISTRY_TAG,
                    image_bytes=image_path.read_bytes(),
                )
                result = MlServiceClient.to_canonical_segment(segment_output)
                summary = SegmentMergeService().apply_sync(
                    session,
                    part_id=job.document_part_id,
                    canonical_segment=result,
                    job_id=job.id,
                )
            mark_job_done(
                job.id,
                {
                    "blocks_count": summary.blocks_count,
                    "lines_count": summary.lines_count,
                    "added_lines": summary.added_lines,
                    "pruned_lines": summary.pruned_lines,
                    "preserved_manual_lines": summary.preserved_manual_lines,
                },
            )
            return
        except Exception as exc:
            logger.exception("Segment job %s failed", job.id, exc_info=exc)
            mark_job_failed(job.id, _public_job_error(exc))
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
