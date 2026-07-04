"""ML worker that waits on Postgres notifications and processes queued jobs."""

from __future__ import annotations

import logging
import time
from threading import Event

from ml.contracts.common import MLJobStatus
from ml.infrastructure.db import JobNotificationListener, engine
from ml.infrastructure.job_repository import (
    claim_next_pending_job,
    mark_job_done,
    mark_job_failed,
    reclaim_stale_running_jobs,
    seconds_until_next_stale_running_job,
)
from ml.infrastructure.settings import get_ml_settings
from ml.jobs.callback import post_job_callback
from ml.jobs.runner import run_job

logger = logging.getLogger(__name__)

_SCHEMA_READY_TIMEOUT_SECONDS = 120.0
_SCHEMA_RETRY_INTERVAL_SECONDS = 2.0


def ml_jobs_table_ready() -> bool:
    from sqlalchemy import inspect

    return inspect(engine).has_table("ml_jobs")


def wait_for_worker_schema(
    *,
    timeout_seconds: float = _SCHEMA_READY_TIMEOUT_SECONDS,
    retry_interval_seconds: float = _SCHEMA_RETRY_INTERVAL_SECONDS,
) -> None:
    """Block until Alembic has created the ml_jobs table."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ml_jobs_table_ready():
            return
        logger.info("waiting for ml_jobs table (platform migrations)")
        time.sleep(retry_interval_seconds)
    raise RuntimeError(
        f"ml_jobs table not found after {timeout_seconds:.0f}s; "
        "run platform migrations before starting ml-worker"
    )


def execute_claimed_job(job) -> None:
    """Run a claimed job and publish its terminal status."""
    try:
        output = run_job(job)
    except Exception as exc:
        logger.exception("ML job %s failed", job.id, exc_info=exc)
        error = "ML job failed"
        mark_job_failed(job.id, error)
        post_job_callback(job, status=MLJobStatus.failed, error=error)
        return

    mark_job_done(job.id, output.model_dump(mode="json"))
    post_job_callback(job, status=MLJobStatus.done, output=output)


def process_next_job() -> bool:
    """Claim and execute at most one pending ML job."""
    job = claim_next_pending_job()
    if job is None:
        return False
    execute_claimed_job(job)
    return True


def run_worker(*, max_jobs: int | None = None, ready_event: Event | None = None) -> None:
    """Run forever, waking on PostgreSQL notifications or stale-job deadlines."""
    wait_for_worker_schema()
    settings = get_ml_settings()
    logger.info("ml-worker listening on PostgreSQL channel %s", settings.worker_notify_channel)

    with JobNotificationListener(settings.worker_notify_channel) as listener:
        processed_jobs = 0
        if ready_event is not None:
            ready_event.set()

        while True:
            reclaimed = reclaim_stale_running_jobs(
                running_timeout_seconds=settings.worker_running_job_timeout_seconds
            )
            if reclaimed:
                logger.warning("reclaimed %s stale ML job(s)", reclaimed)

            if process_next_job():
                processed_jobs += 1
                if max_jobs is not None and processed_jobs >= max_jobs:
                    return
                continue

            wait_timeout = seconds_until_next_stale_running_job(
                running_timeout_seconds=settings.worker_running_job_timeout_seconds
            )
            listener.wait(timeout_seconds=wait_timeout)


def main() -> None:
    try:
        run_worker()
    except Exception:
        logger.exception("ml-worker crashed")
        # Keep the old process shape useful during early deployment; a supervisor
        # should still restart the worker if this keeps happening.
        time.sleep(5)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
