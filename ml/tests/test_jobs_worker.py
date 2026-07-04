"""Worker claim semantics for the ML job queue."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import BytesIO
from threading import Event, Thread
from uuid import uuid4

import pytest
from ml.architectures.mock import mock_segment
from ml.contracts.common import MLJobStatus, MLTask
from ml.contracts.jobs import JobSubmitRequest
from ml.infrastructure.db import JobNotificationListener, SessionLocal
from ml.infrastructure.job_repository import (
    claim_next_pending_job,
    create_job,
    get_job_by_id,
    reclaim_stale_running_jobs,
    seconds_until_next_stale_running_job,
)
from ml.jobs.worker import run_worker, wait_for_worker_schema
from PIL import Image
from sqlalchemy import select

from ml.infrastructure.orm_models import MLJob

pytestmark = pytest.mark.integration


def test_wait_for_worker_schema_raises_when_table_never_appears(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ml.jobs.worker.ml_jobs_table_ready", lambda: False)
    with pytest.raises(RuntimeError, match="ml_jobs table not found"):
        wait_for_worker_schema(timeout_seconds=0.1, retry_interval_seconds=0.01)


def test_wait_for_worker_schema_returns_after_table_appears(monkeypatch: pytest.MonkeyPatch):
    attempts = {"count": 0}

    def ready() -> bool:
        attempts["count"] += 1
        return attempts["count"] >= 2

    monkeypatch.setattr("ml.jobs.worker.ml_jobs_table_ready", ready)
    wait_for_worker_schema(timeout_seconds=1.0, retry_interval_seconds=0.01)
    assert attempts["count"] == 2


def _png(width: int = 1, height: int = 1) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def _submit(product_job_id=None) -> MLJob:
    return create_job(
        JobSubmitRequest(
            task=MLTask.segment,
            registry_model_id="kraken-blla",
            product_job_id=product_job_id or uuid4(),
            image_bytes=_png(),
        )
    )


@pytest.fixture(autouse=True)
def synthetic_worker_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _run_job(job: MLJob):
        if job.task == MLTask.segment:
            return mock_segment(job.image_bytes)
        raise ValueError(f"unsupported test task: {job.task}")

    monkeypatch.setattr("ml.jobs.worker.run_job", _run_job)


def _assert_synthetic_segment_output(output: dict) -> None:
    assert output["blocks"][0]["external_id"] == "kraken-block-1"
    line = output["lines"][0]
    assert line["external_id"] == "kraken-line-1"
    assert line["block_external_id"] == "kraken-block-1"
    assert line["source_metadata"] == {"adapter": "kraken_stub"}


def test_worker_claims_only_pending_jobs():
    job = _submit()
    claimed = claim_next_pending_job()
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == MLJobStatus.running

    second = claim_next_pending_job()
    assert second is None


def test_create_job_notifies_waiting_workers():
    with JobNotificationListener() as listener:
        job = _submit()
        payloads = listener.wait(timeout_seconds=1.0)

    assert str(job.id) in payloads


def test_worker_listens_for_new_jobs_and_processes_notification():
    ready = Event()
    worker = Thread(
        target=run_worker,
        kwargs={"max_jobs": 1, "ready_event": ready},
        daemon=True,
    )
    worker.start()

    assert ready.wait(timeout=5.0)
    job = _submit()

    worker.join(timeout=5.0)
    assert not worker.is_alive()

    processed = get_job_by_id(job.id)
    assert processed is not None
    assert processed.status == MLJobStatus.done
    assert processed.output is not None
    _assert_synthetic_segment_output(processed.output)


def test_reclaim_stale_running_jobs_moves_expired_jobs_to_pending():
    job = _submit()
    claimed = claim_next_pending_job()
    assert claimed is not None
    assert claimed.id == job.id

    with SessionLocal() as session:
        running = session.get(MLJob, job.id)
        assert running is not None
        running.started_at = datetime.now(UTC) - timedelta(minutes=31)
        session.commit()

    reclaimed = reclaim_stale_running_jobs(running_timeout_seconds=1800.0)
    assert reclaimed == 1

    recovered = get_job_by_id(job.id)
    assert recovered is not None
    assert recovered.status == MLJobStatus.pending
    assert recovered.started_at is None


def test_worker_recovers_stale_running_job_without_notification(
    monkeypatch: pytest.MonkeyPatch,
):
    from ml.infrastructure.settings import MLSettings

    settings = MLSettings()
    settings.worker_running_job_timeout_seconds = 0.1
    monkeypatch.setattr("ml.jobs.worker.get_ml_settings", lambda: settings)
    monkeypatch.setattr("ml.infrastructure.db.get_ml_settings", lambda: settings)

    job = _submit()
    claimed = claim_next_pending_job()
    assert claimed is not None

    with SessionLocal() as session:
        running = session.get(MLJob, job.id)
        assert running is not None
        running.started_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()

    assert seconds_until_next_stale_running_job(running_timeout_seconds=0.1) == 0.0

    ready = Event()
    worker = Thread(
        target=run_worker,
        kwargs={"max_jobs": 1, "ready_event": ready},
        daemon=True,
    )
    worker.start()

    assert ready.wait(timeout=5.0)
    worker.join(timeout=5.0)
    assert not worker.is_alive()

    processed = get_job_by_id(job.id)
    assert processed is not None
    assert processed.status == MLJobStatus.done
    assert processed.output is not None
    _assert_synthetic_segment_output(processed.output)


def test_skip_locked_allows_parallel_claims():
    first = _submit()
    second = _submit()

    session = SessionLocal()
    locked = (
        session.execute(
            select(MLJob)
            .where(MLJob.status == MLJobStatus.pending)
            .order_by(MLJob.created_at, MLJob.id)
            .with_for_update()
            .limit(1)
        )
        .scalar_one_or_none()
    )
    assert locked is not None

    claimed = claim_next_pending_job()
    assert claimed is not None
    assert claimed.id in {first.id, second.id}
    assert claimed.id != locked.id

    session.rollback()
    session.close()
