"""Worker claim semantics for the ML job queue."""

from __future__ import annotations

from uuid import uuid4

from ml.contracts.common import MLJobStatus, MLTask
from ml.contracts.jobs import JobSubmitRequest
from ml.infrastructure.db import SessionLocal
from ml.infrastructure.job_repository import claim_next_pending_job, create_job
from sqlalchemy import select

from ml.infrastructure.orm_models import MLJob


def _submit(product_job_id=None) -> MLJob:
    return create_job(
        JobSubmitRequest(
            task=MLTask.segment,
            registry_model_id="kraken-blla",
            product_job_id=product_job_id or uuid4(),
            image_bytes=b"page",
        )
    )


def test_worker_claims_only_pending_jobs():
    job = _submit()
    claimed = claim_next_pending_job()
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == MLJobStatus.running

    second = claim_next_pending_job()
    assert second is None


def test_skip_locked_allows_parallel_claims():
    first = _submit()
    second = _submit()

    session = SessionLocal()
    locked = (
        session.execute(
            select(MLJob)
            .where(MLJob.status == MLJobStatus.pending)
            .order_by(MLJob.created_at)
            .with_for_update()
            .limit(1)
        )
        .scalar_one_or_none()
    )
    assert locked is not None

    claimed = claim_next_pending_job()
    assert claimed is not None
    assert claimed.id == second.id if locked.id == first.id else first.id

    session.rollback()
    session.close()
