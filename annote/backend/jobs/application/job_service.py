"""Job application service — enqueue and read job state."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.jobs.infrastructure.job_repository import JobRepository
from backend.jobs.infrastructure.orm_models import Job


class JobService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = JobRepository(session)

    async def enqueue_test_job(self, handler: str = "noop", *, user_id: uuid.UUID | None = None) -> Job:
        return await self._repo.create_test_job(handler, user_id=user_id)

    async def enqueue_segment_job(
        self,
        *,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        document_part_id: uuid.UUID,
        model_id: uuid.UUID | None = None,
        binding_id: uuid.UUID | None = None,
    ) -> Job:
        return await self._repo.create_segment_job(
            user_id=user_id,
            document_id=document_id,
            document_part_id=document_part_id,
            model_id=model_id,
            binding_id=binding_id,
        )

    async def get_job(self, job_id: uuid.UUID) -> Job:
        job = await self._repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job
