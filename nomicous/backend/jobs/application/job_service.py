"""Job application service — enqueue and read job state."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError
from backend.jobs.infrastructure.job_repository import JobRepository
from backend.jobs.infrastructure.job_repository import cancel_job as cancel_job_row
from backend.jobs.infrastructure.orm_models import Job, JobStatus


class JobService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = JobRepository(session)

    async def enqueue_test_job(
        self, handler: str = "noop", *, user_id: uuid.UUID | None = None
    ) -> Job:
        return await self._repo.create_test_job(handler, user_id=user_id)

    async def get_job(self, job_id: uuid.UUID) -> Job:
        job = await self._repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job

    async def list_project_jobs(
        self,
        project_id: uuid.UUID,
        *,
        limit: int = 50,
        cursor=None,
    ) -> list[Job]:
        return await self._repo.list_for_project(project_id, limit=limit, cursor=cursor)

    async def cancel_job(self, job_id: uuid.UUID) -> Job:
        existing = await self.get_job(job_id)
        if existing.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
            raise ConflictError(
                f"job {job_id} cannot be cancelled from status {existing.status.value}"
            )
        job = cancel_job_row(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job
