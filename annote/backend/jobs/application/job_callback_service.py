"""Apply ML job completion callbacks to Product jobs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from ml_service.contracts.common import MLJobStatus, MLTask
from ml_service.contracts.jobs import JobCallbackRequest
from ml_service.contracts.segment import SegmentRunResponse
from ml_service.contracts.transcribe import TranscribeRunResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError
from backend.document.application.segment_merge_service import SegmentMergeService
from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.document.infrastructure.orm_models import Line
from backend.jobs.infrastructure.job_repository import JobRepository
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.ml.infrastructure.ml_client import MlServiceClient
from infrastructure.db import SyncSessionLocal

_TERMINAL_STATUSES = frozenset({JobStatus.done, JobStatus.failed})


def _job_type_for_task(task: MLTask) -> JobType:
    return JobType(task.value)


def _serialize_callback_result(callback: JobCallbackRequest) -> dict:
    output = callback.output.model_dump(mode="json") if callback.output is not None else None
    return {
        "ml_job_id": str(callback.ml_job_id),
        "task": callback.task.value,
        "output": output,
    }


def _known_ml_job_ids(job: Job) -> set[uuid.UUID]:
    ids: set[uuid.UUID] = set()
    if job.ml_job_id is not None:
        ids.add(job.ml_job_id)
    for entry in (job.payload or {}).get("ml_line_jobs", []):
        ids.add(uuid.UUID(str(entry["ml_job_id"])))
    return ids


def _segment_output(callback: JobCallbackRequest) -> SegmentRunResponse:
    if callback.output is None or callback.output.kind != "segment":
        raise ValueError("Segment callback missing structured output")
    return callback.output.data


def _transcribe_output(callback: JobCallbackRequest) -> TranscribeRunResponse:
    if callback.output is None or callback.output.kind != "transcribe":
        raise TranscribeJobHandlerError("Transcribe callback missing structured output")
    return callback.output.data


def _apply_segment_merge(job: Job, callback: JobCallbackRequest) -> dict:
    if job.document_part_id is None:
        return _serialize_callback_result(callback)
    canonical = MlServiceClient.to_canonical_segment(_segment_output(callback))
    with SyncSessionLocal() as session:
        summary = SegmentMergeService().apply_sync(
            session,
            part_id=job.document_part_id,
            canonical_segment=canonical,
            job_id=job.id,
        )
    return {
        "blocks_count": summary.blocks_count,
        "lines_count": summary.lines_count,
        "added_lines": summary.added_lines,
        "pruned_lines": summary.pruned_lines,
        "preserved_manual_lines": summary.preserved_manual_lines,
    }


def _transcribe_line_jobs_and_outputs(
    payload: dict,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return (
        list(payload.get("ml_line_jobs", [])),
        dict(payload.get("ml_line_outputs", {})),
    )


def _apply_transcribe_merge_sync(
    session,
    *,
    job: Job,
    line_jobs: list[dict[str, object]],
    line_outputs: dict[str, object],
) -> dict:
    if job.document_id is None or job.document_part_id is None:
        raise TranscribeJobHandlerError("Transcribe job is missing its target document part")

    lines_with_output: list[tuple[Line, TranscribeRunResponse]] = []
    for entry in sorted(line_jobs, key=lambda item: int(item["line_index"])):
        line = session.get(Line, uuid.UUID(str(entry["line_id"])))
        if line is None:
            raise TranscribeJobHandlerError("Document line not found")
        output = TranscribeRunResponse.model_validate(
            line_outputs[str(entry["ml_job_id"])]
        )
        lines_with_output.append((line, output))
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
) -> tuple[bool, dict | None]:
    if job.document_id is None or job.document_part_id is None:
        return True, _serialize_callback_result(callback)

    payload = dict(job.payload or {})
    line_jobs, line_outputs = _transcribe_line_jobs_and_outputs(payload)
    line_outputs[str(callback.ml_job_id)] = _transcribe_output(callback).model_dump(mode="json")
    payload["ml_line_outputs"] = line_outputs

    if not line_jobs:
        return False, payload

    if len(line_outputs) < len(line_jobs):
        return False, payload

    result = _apply_transcribe_merge_sync(
        session,
        job=job,
        line_jobs=line_jobs,
        line_outputs=line_outputs,
    )
    return True, result


def _apply_transcribe_callback_locked(callback: JobCallbackRequest) -> bool:
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
        if job.type != JobType.transcribe:
            raise ConflictError(
                f"job {job.id} type {job.type.value} "
                f"does not match callback task {callback.task.value}"
            )

        known_ids = _known_ml_job_ids(job)
        if known_ids and callback.ml_job_id not in known_ids:
            raise ConflictError(
                f"job {job.id} does not recognize callback ml_job_id {callback.ml_job_id}"
            )

        completed, payload_or_result = _apply_transcribe_callback(session, job, callback)
        now = datetime.now(UTC)
        if not completed:
            assert isinstance(payload_or_result, dict)
            job.payload = payload_or_result
            job.updated_at = now
            session.commit()
            return True

        job.status = JobStatus.done
        job.ml_job_id = callback.ml_job_id
        job.result = payload_or_result or {}
        job.error = None
        job.completed_at = now
        job.updated_at = now
        session.commit()
        return True


def try_complete_transcribe_job(job_id: uuid.UUID) -> bool:
    """Complete a transcribe job when all ML line callbacks were buffered early."""
    with SyncSessionLocal() as session:
        job = session.execute(
            select(Job).where(Job.id == job_id).with_for_update()
        ).scalar_one_or_none()
        if job is None or job.type != JobType.transcribe:
            return False
        if job.status in _TERMINAL_STATUSES:
            return False
        payload = dict(job.payload or {})
        line_jobs, line_outputs = _transcribe_line_jobs_and_outputs(payload)
        if not line_jobs or len(line_outputs) < len(line_jobs):
            return False
        if job.document_id is None or job.document_part_id is None:
            return False

        result = _apply_transcribe_merge_sync(
            session,
            job=job,
            line_jobs=line_jobs,
            line_outputs=line_outputs,
        )
        now = datetime.now(UTC)
        job.status = JobStatus.done
        job.result = result
        job.error = None
        job.completed_at = now
        job.updated_at = now
        session.commit()
        return True


class JobCallbackService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = JobRepository(session)

    async def apply_callback(self, callback: JobCallbackRequest) -> bool:
        """Apply callback. Returns False when the Product job was already terminal."""
        job = await self._repo.get_by_id(callback.product_job_id)
        if job is None:
            raise NotFoundError(f"job {callback.product_job_id} not found")

        if job.status in _TERMINAL_STATUSES:
            return False

        expected_type = _job_type_for_task(callback.task)
        if job.type != expected_type:
            raise ConflictError(
                f"job {job.id} type {job.type.value} "
                f"does not match callback task {callback.task.value}"
            )

        known_ids = _known_ml_job_ids(job)
        if known_ids and callback.ml_job_id not in known_ids:
            raise ConflictError(
                f"job {job.id} does not recognize callback ml_job_id {callback.ml_job_id}"
            )

        if callback.status == MLJobStatus.failed:
            await self._repo.mark_failed_from_callback(
                job.id,
                ml_job_id=callback.ml_job_id,
                error=callback.error or "ML job failed",
            )
            return True

        if job.type == JobType.segment:
            result = _apply_segment_merge(job, callback)
            await self._repo.mark_done_from_callback(
                job.id,
                ml_job_id=callback.ml_job_id,
                result=result,
            )
            return True

        if job.type == JobType.transcribe:
            return _apply_transcribe_callback_locked(callback)

        await self._repo.mark_done_from_callback(
            job.id,
            ml_job_id=callback.ml_job_id,
            result=_serialize_callback_result(callback),
        )
        return True
