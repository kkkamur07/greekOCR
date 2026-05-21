"""Job application service — enqueue and read job state."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.inference.infrastructure.job_repository import JobRepository
from backend.inference.infrastructure.orm_models import Job


class JobService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = JobRepository(session)

    async def enqueue_test_job(self, handler: str = "noop") -> Job:
        return await self._repo.create_test_job(handler)

    async def get_job(self, job_id: uuid.UUID) -> Job:
        job = await self._repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job
