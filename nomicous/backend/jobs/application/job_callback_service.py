"""Apply inference job completion callbacks to Product jobs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from inference.contracts.common import InferenceJobStatus, InferenceTask as WireInferenceTask
from inference.contracts.jobs import JobCallbackRequest
from inference.contracts.segment import SegmentRunResponse
from inference.contracts.transcribe import TranscribeBatchRunResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError
from backend.document.application.segment_merge_service import SegmentMergeService
from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.document.infrastructure.orm_models import Line
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.infrastructure.ml_client import InferenceClient
from infrastructure.db import SyncSessionLocal

_TERMINAL_STATUSES = frozenset({JobStatus.done, JobStatus.failed})


def _job_type_for_task(task: WireInferenceTask) -> JobType:
    return JobType(task.value)


def _known_inference_job_ids(job: Job) -> set[uuid.UUID]:
    ids: set[uuid.UUID] = set()
    if job.inference_job_id is not None:
        ids.add(job.inference_job_id)
    return ids


def _segment_output(callback: JobCallbackRequest) -> SegmentRunResponse:
    if callback.output is None or callback.output.kind != "segment":
        raise ValueError("Segment callback missing structured output")
    return callback.output.data


def _transcribe_output(callback: JobCallbackRequest) -> TranscribeBatchRunResponse:
    if callback.output is None or callback.output.kind != "transcribe":
        raise TranscribeJobHandlerError("Transcribe callback missing structured output")
    data = callback.output.data
    if not isinstance(data, TranscribeBatchRunResponse):
        raise TranscribeJobHandlerError("Transcribe callback missing batched line results")
    return data


def _apply_segment_merge(session, job: Job, callback: JobCallbackRequest) -> dict:
    if job.document_part_id is None:
        raise ValueError("Segment job is missing its target document part")
    canonical = InferenceClient.to_canonical_segment(_segment_output(callback))
    summary = SegmentMergeService().apply_sync(
        session,
        part_id=job.document_part_id,
        canonical_segment=canonical,
        job_id=job.id,
        commit=False,
    )
    return {
        "blocks_count": summary.blocks_count,
        "lines_count": summary.lines_count,
        "added_lines": summary.added_lines,
        "pruned_lines": summary.pruned_lines,
        "preserved_manual_lines": summary.preserved_manual_lines,
    }


def _apply_transcribe_merge_sync(
    session,
    *,
    job: Job,
    output: TranscribeBatchRunResponse,
) -> dict:
    if job.document_id is None or job.document_part_id is None:
        raise TranscribeJobHandlerError("Transcribe job is missing its target document part")

    lines_with_output = []
    for result in sorted(output.lines, key=lambda item: item.line_index):
        if result.line_id is None:
            raise TranscribeJobHandlerError("Transcribe callback line is missing line_id")
        try:
            line_id = uuid.UUID(result.line_id)
        except ValueError as exc:
            raise TranscribeJobHandlerError("Transcribe callback line_id is invalid") from exc
        line = session.get(Line, line_id)
        if line is None or line.part_id != job.document_part_id:
            raise TranscribeJobHandlerError("Document line not found")
        lines_with_output.append((line, result.output))
    return TranscribeMergeService().apply_sync(
        session,
        document_id=job.document_id,
        part_id=job.document_part_id,
        job_id=job.id,
        lines_with_output=lines_with_output,
        commit=False,
    )


def _apply_transcribe_callback(
    session,
    job: Job,
    callback: JobCallbackRequest,
) -> dict:
    if job.document_id is None or job.document_part_id is None:
        raise TranscribeJobHandlerError("Transcribe job is missing its target document part")

    output = _transcribe_output(callback)
    return _apply_transcribe_merge_sync(
        session,
        job=job,
        output=output,
    )


def _mark_failed_from_callback_sync(job: Job, callback: JobCallbackRequest) -> None:
    now = datetime.now(UTC)
    job.status = JobStatus.failed
    job.inference_job_id = callback.inference_job_id
    job.error = callback.error or "Inference job failed"
    job.completed_at = now
    job.updated_at = now


def _assert_callback_matches_job(job: Job, callback: JobCallbackRequest) -> None:
    expected_type = _job_type_for_task(callback.task)
    if job.type != expected_type:
        raise ConflictError(
            f"job {job.id} type {job.type.value} "
            f"does not match callback task {callback.task.value}"
        )

    known_ids = _known_inference_job_ids(job)
    if known_ids and callback.inference_job_id not in known_ids:
        raise ConflictError(
            f"job {job.id} does not recognize callback inference_job_id {callback.inference_job_id}"
        )


def _mark_done_from_callback_sync(job: Job, callback: JobCallbackRequest, result: dict) -> None:
    now = datetime.now(UTC)
    job.status = JobStatus.done
    job.inference_job_id = callback.inference_job_id
    job.result = result
    job.error = None
    job.completed_at = now
    job.updated_at = now


def _apply_callback_locked(callback: JobCallbackRequest) -> bool:
    with SyncSessionLocal() as session:
        job = session.execute(
            select(Job)
            .where(Job.id == callback.product_job_id)
            .with_for_update()
        ).scalar_one_or_none()
        if job is None:
            raise NotFoundError(f"job {callback.product_job_id} not found")
        if job.status in _TERMINAL_STATUSES:
            return False
        _assert_callback_matches_job(job, callback)

        if callback.status == InferenceJobStatus.failed:
            _mark_failed_from_callback_sync(job, callback)
            session.commit()
            notify_platform_job_status_changed(job.id, job.status)
            return True

        if job.type == JobType.segment:
            result = _apply_segment_merge(session, job, callback)
        elif job.type == JobType.transcribe:
            result = _apply_transcribe_callback(session, job, callback)
        else:
            raise ConflictError(f"job {job.id} type {job.type.value} cannot receive inference callbacks")

        _mark_done_from_callback_sync(job, callback, result)
        session.commit()
        notify_platform_job_status_changed(job.id, job.status)
        return True


class JobCallbackService:
    def __init__(self, _session: AsyncSession) -> None:
        pass

    async def apply_callback(self, callback: JobCallbackRequest) -> bool:
        """Apply callback. Returns False when the Product job was already terminal."""
        return _apply_callback_locked(callback)
