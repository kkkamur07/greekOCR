"""Apply ML job completion callbacks to Product jobs."""

from __future__ import annotations

from ml.contracts.common import MLJobStatus, MLTask
from ml.contracts.jobs import JobCallbackRequest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError
from backend.jobs.infrastructure.job_repository import JobRepository
from backend.jobs.infrastructure.orm_models import JobStatus, JobType

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

        if job.ml_job_id is not None and job.ml_job_id != callback.ml_job_id:
            raise ConflictError(
                f"job {job.id} ml_job_id {job.ml_job_id} "
                f"does not match callback {callback.ml_job_id}"
            )

        if callback.status == MLJobStatus.done:
            await self._repo.mark_done_from_callback(
                job.id,
                ml_job_id=callback.ml_job_id,
                result=_serialize_callback_result(callback),
            )
            return True

        await self._repo.mark_failed_from_callback(
            job.id,
            ml_job_id=callback.ml_job_id,
            error=callback.error or "ML job failed",
        )
        return True
