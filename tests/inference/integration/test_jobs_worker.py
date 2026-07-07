"""Worker claim semantics for the ML job queue."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Event, Thread
from uuid import uuid4

import pytest
from inference.contracts.common import InferenceJobStatus, InferenceTask
from inference.contracts.jobs import JobSubmitRequest
from inference.infrastructure.db import JobNotificationListener, SessionLocal
from inference.infrastructure.job_repository import (
    claim_next_pending_job,
    create_job,
    get_job_by_id,
    reclaim_stale_running_jobs,
    seconds_until_next_stale_running_job,
)
from inference.infrastructure.orm_models import InferenceJob
from inference.jobs.worker import run_worker, wait_for_worker_schema
from sqlalchemy import select
from tests.fixtures.paths import segment_page_bytes, transcribe_line_bytes

pytestmark = pytest.mark.integration


# --- Worker startup: wait for inference_jobs table ---
# Tests schema polling and timeout. Does not run migrations or start a worker.


def test_wait_for_worker_schema_raises_when_table_never_appears(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("inference.jobs.worker.inference_jobs_table_ready", lambda: False)
    with pytest.raises(RuntimeError, match="inference_jobs table not found"):
        wait_for_worker_schema(timeout_seconds=0.1, retry_interval_seconds=0.01)


def test_wait_for_worker_schema_returns_after_table_appears(monkeypatch: pytest.MonkeyPatch):
    attempts = {"count": 0}

    def ready() -> bool:
        attempts["count"] += 1
        return attempts["count"] >= 2

    monkeypatch.setattr("inference.jobs.worker.inference_jobs_table_ready", ready)
    wait_for_worker_schema(timeout_seconds=1.0, retry_interval_seconds=0.01)
    assert attempts["count"] == 2


def _submit_segment(product_job_id=None) -> InferenceJob:
    # Full-page manuscript fixture; Kraken returns multiple text lines.
    return create_job(
        JobSubmitRequest(
            task=InferenceTask.segment,
            registry_model_id="kraken-blla",
            product_job_id=product_job_id or uuid4(),
            image_bytes=segment_page_bytes(),
        )
    )


def _submit_transcribe(product_job_id=None) -> InferenceJob:
    # Single pre-cropped line image; one batched Calamari region (line_index 0).
    return create_job(
        JobSubmitRequest(
            task=InferenceTask.transcribe,
            registry_model_id="syriac-calamariv1",
            product_job_id=product_job_id or uuid4(),
            image_bytes=transcribe_line_bytes(),
            params={"lines": [{"line_index": 0}]},
        )
    )


def _start_worker(*, max_jobs: int) -> tuple[Thread, Event]:
    ready = Event()
    worker = Thread(
        target=run_worker,
        kwargs={"max_jobs": max_jobs, "ready_event": ready},
    )
    worker.start()
    assert ready.wait(timeout=5.0)
    return worker, ready


# --- Job claiming ---
# Tests claim moves pending -> running and only one job is claimed. Does not execute inference.


def test_worker_claims_only_pending_jobs():
    job = _submit_segment()
    claimed = claim_next_pending_job()
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == InferenceJobStatus.running

    second = claim_next_pending_job()
    assert second is None


# --- Stale job reclaim (repository) ---
# Tests expired running jobs go back to pending. Does not start a worker or run inference.


def test_reclaim_stale_running_jobs_moves_expired_jobs_to_pending():
    job = _submit_segment()
    claimed = claim_next_pending_job()
    assert claimed is not None
    assert claimed.id == job.id

    with SessionLocal() as session:
        running = session.get(InferenceJob, job.id)
        assert running is not None
        running.started_at = datetime.now(UTC) - timedelta(minutes=31)
        session.commit()

    reclaimed = reclaim_stale_running_jobs(running_timeout_seconds=1800.0)
    assert reclaimed == 1

    recovered = get_job_by_id(job.id)
    assert recovered is not None
    assert recovered.status == InferenceJobStatus.pending
    assert recovered.started_at is None


# --- Parallel claims (SKIP LOCKED) ---
# Tests a second worker can claim while another row is locked. Does not execute inference.


def test_skip_locked_allows_parallel_claims():
    first = _submit_segment()
    second = _submit_segment()

    session = SessionLocal()
    locked = (
        session.execute(
            select(InferenceJob)
            .where(InferenceJob.status == InferenceJobStatus.pending)
            .order_by(InferenceJob.created_at, InferenceJob.id)
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


# --- Worker thread end-to-end (ML lane) ---
# Tests pg_notify on create_job, worker LISTEN wakeup, Kraken page segment, one-line Calamari transcribe.


@pytest.mark.ml
def test_worker_processes_segment_and_transcribe_jobs_on_notify():
    worker, _ready = _start_worker(max_jobs=2)

    with JobNotificationListener() as listener:
        segment_job = _submit_segment()
        assert str(segment_job.id) in listener.wait(timeout_seconds=5.0)

        transcribe_job = _submit_transcribe()
        assert str(transcribe_job.id) in listener.wait(timeout_seconds=5.0)

    worker.join(timeout=120.0)
    assert not worker.is_alive()

    segment_processed = get_job_by_id(segment_job.id)
    assert segment_processed is not None
    assert segment_processed.status == InferenceJobStatus.done
    assert segment_processed.output is not None
    assert len(segment_processed.output["lines"]) > 1
    assert segment_processed.output["lines"][0]["source_metadata"]["adapter"] == "kraken"

    transcribe_processed = get_job_by_id(transcribe_job.id)
    assert transcribe_processed is not None
    assert transcribe_processed.status == InferenceJobStatus.done
    assert transcribe_processed.output is not None
    assert len(transcribe_processed.output["lines"]) == 1
    assert transcribe_processed.output["lines"][0]["line_index"] == 0
    assert isinstance(transcribe_processed.output["lines"][0]["output"]["text"], str)


# --- Worker stale reclaim without NOTIFY (ML lane) ---
# Tests worker polls, reclaims a stuck running job, and runs Kraken. Does not test NOTIFY wakeup.


@pytest.mark.ml
def test_worker_recovers_stale_running_job_without_notification(
    monkeypatch: pytest.MonkeyPatch,
):
    from inference.infrastructure.settings import InferenceSettings

    settings = InferenceSettings()
    settings.worker_running_job_timeout_seconds = 0.1
    monkeypatch.setattr("inference.jobs.worker.get_inference_settings", lambda: settings)
    monkeypatch.setattr("inference.infrastructure.db.get_inference_settings", lambda: settings)

    job = _submit_segment()
    claimed = claim_next_pending_job()
    assert claimed is not None

    with SessionLocal() as session:
        running = session.get(InferenceJob, job.id)
        assert running is not None
        running.started_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()

    assert seconds_until_next_stale_running_job(running_timeout_seconds=0.1) == 0.0

    worker, _ready = _start_worker(max_jobs=1)
    worker.join(timeout=120.0)
    assert not worker.is_alive()

    processed = get_job_by_id(job.id)
    assert processed is not None
    assert processed.status == InferenceJobStatus.done
    assert processed.output is not None
    assert processed.output["lines"][0]["source_metadata"]["adapter"] == "kraken"
