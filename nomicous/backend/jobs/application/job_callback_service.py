"""Apply inference job completion callbacks to Product jobs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
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
from infrastructure.db import sync_system_session

_TERMINAL_STATUSES = frozenset({JobStatus.done, JobStatus.failed, JobStatus.cancelled})


@dataclass(frozen=True)
class _MergeContext:
    job_id: uuid.UUID
    job_type: JobType
    document_id: uuid.UUID | None
    document_part_id: uuid.UUID | None
    inference_job_id: uuid.UUID


def _job_type_for_task(task: WireInferenceTask) -> JobType:
    return JobType(task.value)


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


def _merge_context(job: Job, callback: JobCallbackRequest) -> _MergeContext:
    return _MergeContext(
        job_id=job.id,
        job_type=job.type,
        document_id=job.document_id,
        document_part_id=job.document_part_id,
        inference_job_id=callback.inference_job_id,
    )


def _apply_segment_merge(session, context: _MergeContext, callback: JobCallbackRequest) -> dict:
    if context.document_part_id is None:
        raise ValueError("Segment job is missing its target document part")
    canonical = InferenceClient.to_canonical_segment(_segment_output(callback))
    summary = SegmentMergeService().apply_sync(
        session,
        part_id=context.document_part_id,
        canonical_segment=canonical,
        job_id=context.job_id,
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
    context: _MergeContext,
    output: TranscribeBatchRunResponse,
) -> dict:
    if context.document_id is None or context.document_part_id is None:
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
        if line is None or line.part_id != context.document_part_id:
            raise TranscribeJobHandlerError("Document line not found")
        lines_with_output.append((line, result.output))
    return TranscribeMergeService().apply_sync(
        session,
        document_id=context.document_id,
        part_id=context.document_part_id,
        job_id=context.job_id,
        lines_with_output=lines_with_output,
        commit=False,
    )


def _run_merge(context: _MergeContext, callback: JobCallbackRequest) -> dict:
    with sync_system_session() as session:
        if context.job_type == JobType.segment:
            result = _apply_segment_merge(session, context, callback)
        elif context.job_type == JobType.transcribe:
            result = _apply_transcribe_merge_sync(
                session,
                context=context,
                output=_transcribe_output(callback),
            )
        else:
            raise ConflictError(
                f"job {context.job_id} type {context.job_type.value} cannot receive inference callbacks"
            )
        session.commit()
        return result


def _mark_failed_from_callback_sync(job: Job, callback: JobCallbackRequest) -> None:
    now = datetime.now(UTC)
    job.status = JobStatus.failed
    job.error = "Inference job failed"
    job.callback_claimed_at = None
    job.completed_at = now
    job.updated_at = now


def _assert_callback_matches_job(job: Job, callback: JobCallbackRequest) -> None:
    expected_type = _job_type_for_task(callback.task)
    if job.type != expected_type:
        raise ConflictError(
            f"job {job.id} type {job.type.value} does not match callback task {callback.task.value}"
        )

    if job.inference_job_id is None or callback.inference_job_id != job.inference_job_id:
        raise ConflictError(f"job {job.id} does not recognize callback inference_job_id")


def _mark_done_from_callback_sync(
    job: Job,
    callback: JobCallbackRequest,
    result: dict,
) -> None:
    now = datetime.now(UTC)
    job.status = JobStatus.done
    job.result = result
    job.error = None
    job.callback_claimed_at = None
    job.completed_at = now
    job.updated_at = now


def _mark_failed_after_merge(job_id: uuid.UUID, error: str) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        job = session.get(Job, job_id)
        if job is None or job.status != JobStatus.waiting or job.callback_claimed_at is None:
            return
        job.status = JobStatus.failed
        job.error = error
        job.completed_at = now
        job.updated_at = now
        session.commit()
        notify_platform_job_status_changed(job.id, job.status)


def _validate_callback(callback: JobCallbackRequest) -> tuple[bool, _MergeContext | None]:
    """Atomically claim one waiting callback before its merge can begin."""
    with sync_system_session() as session:
        job = session.execute(
            select(Job).where(Job.id == callback.product_job_id).with_for_update()
        ).scalar_one_or_none()
        if job is None:
            raise NotFoundError(f"job {callback.product_job_id} not found")
        _assert_callback_matches_job(job, callback)
        if job.status in _TERMINAL_STATUSES:
            return False, None
        if job.status != JobStatus.waiting:
            raise ConflictError(f"job {job.id} is not waiting for an inference callback")
        if job.callback_claimed_at is not None:
            return False, None

        if callback.status == InferenceJobStatus.failed:
            _mark_failed_from_callback_sync(job, callback)
            session.commit()
            notify_platform_job_status_changed(job.id, job.status)
            return True, None

        context = _merge_context(job, callback)
        job.callback_claimed_at = datetime.now(UTC)
        job.updated_at = job.callback_claimed_at
        session.commit()
        return True, context


def _finalize_successful_callback(
    context: _MergeContext, callback: JobCallbackRequest, result: dict
) -> bool:
    with sync_system_session() as session:
        job = session.execute(
            select(Job).where(Job.id == context.job_id).with_for_update()
        ).scalar_one_or_none()
        if job is None:
            raise NotFoundError(f"job {context.job_id} not found")
        _assert_callback_matches_job(job, callback)
        if job.status in _TERMINAL_STATUSES:
            return False
        if job.status != JobStatus.waiting or job.callback_claimed_at is None:
            raise ConflictError(f"job {job.id} is not processing this inference callback")
        _mark_done_from_callback_sync(job, callback, result)
        session.commit()
        notify_platform_job_status_changed(job.id, job.status)
        return True


def _apply_callback_locked(callback: JobCallbackRequest) -> bool:
    applied, context = _validate_callback(callback)
    if not applied or context is None:
        return applied

    try:
        result = _run_merge(context, callback)
    except Exception:
        _mark_failed_after_merge(context.job_id, "Callback processing failed")
        raise

    return _finalize_successful_callback(context, callback, result)


class JobCallbackService:
    def __init__(self, _session: AsyncSession) -> None:
        pass

    async def apply_callback(self, callback: JobCallbackRequest) -> bool:
        """Apply callback. Returns False when the Product job was already terminal."""
        return _apply_callback_locked(callback)
