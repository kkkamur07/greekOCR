"""Background job worker — polls Postgres and runs claimed jobs.

Multiple Uvicorn workers are safe: claim uses ``FOR UPDATE SKIP LOCKED`` so only
one process runs a given job. Parallelism here means *different* jobs at once,
not the same job twice.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from ml_service.contracts.common import MLTask
from ml_service.contracts.jobs import JobSubmitRequest
from PIL import Image

from backend.core.settings.job import get_job_settings
from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.document.infrastructure.media_store import MediaStore
from backend.document.infrastructure.orm_models import DocumentPart, Line
from backend.jobs.infrastructure.handlers import TestJobHandlerError, run_test_handler
from backend.jobs.infrastructure.job_repository import (
    claim_next_pending_job,
    mark_job_done,
    mark_job_failed,
    mark_job_waiting,
)
from backend.jobs.infrastructure.orm_models import Job, JobType
from backend.ml.infrastructure.ml_client import MlServiceClient
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from infrastructure.db import SyncSessionLocal

if TYPE_CHECKING:
    from asyncio import Event

logger = logging.getLogger(__name__)

_DEFAULT_SEGMENT_REGISTRY_MODEL = "kraken-blla"
_DEFAULT_SEGMENT_REGISTRY_TAG = "stable"
_DEFAULT_TRANSCRIBE_REGISTRY_MODEL = "syriac-calamariv1"
_DEFAULT_TRANSCRIBE_REGISTRY_TAG = "stable"
_ml_client: MlServiceClient | None = None


@dataclass(frozen=True)
class RegistrySelection:
    model_id: str
    tag: str


def _get_ml_client() -> MlServiceClient:
    global _ml_client
    if _ml_client is None:
        _ml_client = MlServiceClient()
    return _ml_client


def _public_job_error(exc: BaseException, *, fallback: str = "Job failed") -> str:
    """User-safe message for Job.error; full detail goes to logs only."""
    if isinstance(exc, (TestJobHandlerError, TranscribeJobHandlerError, ValueError)):
        return str(exc)
    return fallback


def _registry_selection_from_artifact_ref(artifact_ref: str) -> RegistrySelection:
    parsed = urlparse(artifact_ref)

    if parsed.scheme != "registry" or not parsed.netloc or parsed.path not in ("", "/"):
        raise ValueError(
            "Inference model artifact_ref must be registry://<registry_model_id>?tag=<tag>"
        )

    tag = parse_qs(parsed.query).get("tag", ["stable"])[0] or "stable"
    return RegistrySelection(model_id=parsed.netloc, tag=tag)


def _job_registry_selection(
    session,
    job: Job,
    *,
    task: InferenceTask,
    fallback_model_id: str,
    fallback_tag: str,
) -> RegistrySelection:

    if job.model_id is None:
        return RegistrySelection(model_id=fallback_model_id, tag=fallback_tag)

    model = session.get(InferenceModel, job.model_id)

    if model is None:
        raise ValueError("Selected inference model not found")
    if model.task != task:
        raise ValueError(f"Selected inference model does not support {task.value}")
        
    return _registry_selection_from_artifact_ref(model.artifact_ref)


def _crop_line_image(image_bytes: bytes, line: Line) -> bytes:
    """Crop the stored page image to the line polygon bounding box."""
    if not line.points:
        return image_bytes

    xs = [point[0] for point in line.points if len(point) == 2]
    ys = [point[1] for point in line.points if len(point) == 2]
    if not xs or not ys:
        return image_bytes

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        left = max(0, int(min(xs)))
        top = max(0, int(min(ys)))
        right = min(width, int(max(xs)))
        bottom = min(height, int(max(ys)))
        if right <= left or bottom <= top:
            return image_bytes

        cropped = image.crop((left, top, right, bottom))
        output = BytesIO()
        cropped.save(output, format=image.format or "PNG")
        return output.getvalue()


def execute_claimed_job(job: Job) -> None:
    """Run handler for a job already in ``running`` status."""
    if job.type == JobType.transcribe:
        try:
            if job.document_id is None or job.document_part_id is None:
                raise TranscribeJobHandlerError(
                    "Transcribe job is missing its target document part"
                )
            with SyncSessionLocal() as session:
                part = session.get(DocumentPart, job.document_part_id)
                if part is None:
                    raise TranscribeJobHandlerError("Document part not found")
                if part.document_id != job.document_id:
                    raise TranscribeJobHandlerError("Document part not found")
                selection = _job_registry_selection(
                    session,
                    job,
                    task=InferenceTask.transcribe,
                    fallback_model_id=_DEFAULT_TRANSCRIBE_REGISTRY_MODEL,
                    fallback_tag=_DEFAULT_TRANSCRIBE_REGISTRY_TAG,
                )
                lines = TranscribeMergeService.load_lines(session, part.id)
                payload = job.payload or {}
                selected_line_ids = payload.get("line_ids")
                if selected_line_ids:
                    allowed = {uuid.UUID(str(line_id)) for line_id in selected_line_ids}
                    lines = [line for line in lines if line.id in allowed]
                    if not lines:
                        raise TranscribeJobHandlerError("No matching lines to transcribe")
                image_bytes = MediaStore().absolute_path(part.image_key).read_bytes()
                line_jobs = []
                base_params = dict((job.payload or {}).get("ml_params") or {})
                for index, line in enumerate(lines):
                    line_image_bytes = _crop_line_image(image_bytes, line)
                    params = {**base_params, "line_index": index, "line_id": str(line.id)}
                    ml_job = _get_ml_client().submit_job(
                        JobSubmitRequest(
                            task=MLTask.transcribe,
                            registry_model_id=selection.model_id,
                            registry_tag=selection.tag,
                            product_job_id=job.id,
                            image_bytes=line_image_bytes,
                            params=params,
                        )
                    )
                    line_jobs.append(
                        {
                            "ml_job_id": str(ml_job.ml_job_id),
                            "line_id": str(line.id),
                            "line_index": index,
                        }
                    )
            mark_job_waiting(
                job.id,
                payload_patch={"ml_line_jobs": line_jobs, "ml_line_outputs": {}},
            )
        except TranscribeJobHandlerError as exc:
            logger.warning("Transcribe job %s failed: %s", job.id, exc)
            mark_job_failed(job.id, _public_job_error(exc))
        except Exception as exc:
            logger.exception("Transcribe job %s failed", job.id, exc_info=exc)
            mark_job_failed(job.id, _public_job_error(exc))
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
                params = dict((job.payload or {}).get("ml_params") or {})
                ml_job = _get_ml_client().submit_job(
                    JobSubmitRequest(
                        task=MLTask.segment,
                        registry_model_id=_DEFAULT_SEGMENT_REGISTRY_MODEL,
                        registry_tag=_DEFAULT_SEGMENT_REGISTRY_TAG,
                        product_job_id=job.id,
                        image_bytes=image_path.read_bytes(),
                        params=params,
                    )
                )
            mark_job_waiting(
                job.id,
                ml_job_id=ml_job.ml_job_id,
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
