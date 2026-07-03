"""POST terminal ML job status to nomicous with retries."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ml.contracts.common import MLJobStatus, MLTask
from ml.contracts.jobs import JobCallbackRequest
from ml.contracts.segment import SegmentRunResponse
from ml.contracts.transcribe import TranscribeRunResponse
from ml.contracts.webhooks import ML_WEBHOOK_SECRET_HEADER
from ml.infrastructure.orm_models import MLJob
from ml.infrastructure.settings import MLSettings

logger = logging.getLogger(__name__)

CALLBACK_MAX_ATTEMPTS = 4
CALLBACK_RETRY_DELAY_SECONDS = 0.5


def _build_callback_payload(
    job: MLJob,
    *,
    status: MLJobStatus,
    output: SegmentRunResponse | TranscribeRunResponse | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    callback = JobCallbackRequest(
        ml_job_id=job.id,
        product_job_id=job.product_job_id,
        task=MLTask(job.task),
        status=status,
        output=output,
        error=error,
    )
    return callback.model_dump(mode="json")


def post_job_callback(
    job: MLJob,
    *,
    status: MLJobStatus,
    output: SegmentRunResponse | TranscribeRunResponse | None = None,
    error: str | None = None,
    settings: MLSettings | None = None,
    client: httpx.Client | None = None,
) -> bool:
    from ml.infrastructure.settings import get_ml_settings

    resolved_settings = settings or get_ml_settings()
    if not resolved_settings.ml_callback_url:
        logger.error(
            "ML_CALLBACK_URL is not configured; skipping callback for ml_job_id=%s",
            job.id,
        )
        return False
    if not resolved_settings.ml_webhook_secret:
        logger.error(
            "ML_WEBHOOK_SECRET is not configured; skipping callback for ml_job_id=%s",
            job.id,
        )
        return False

    payload = _build_callback_payload(job, status=status, output=output, error=error)
    headers = {ML_WEBHOOK_SECRET_HEADER: resolved_settings.ml_webhook_secret}
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)

    try:
        for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
            try:
                response = http.post(
                    resolved_settings.ml_callback_url,
                    json=payload,
                    headers=headers,
                )
                if response.status_code in (200, 204):
                    logger.info(
                        "callback delivered for ml_job_id=%s product_job_id=%s status=%s",
                        job.id,
                        job.product_job_id,
                        status.value,
                    )
                    return True
                logger.warning(
                    "callback attempt %s/%s failed for ml_job_id=%s: HTTP %s",
                    attempt,
                    CALLBACK_MAX_ATTEMPTS,
                    job.id,
                    response.status_code,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "callback attempt %s/%s failed for ml_job_id=%s: %s",
                    attempt,
                    CALLBACK_MAX_ATTEMPTS,
                    job.id,
                    exc,
                )
            if attempt < CALLBACK_MAX_ATTEMPTS:
                time.sleep(CALLBACK_RETRY_DELAY_SECONDS)
        logger.error(
            "callback exhausted retries for ml_job_id=%s product_job_id=%s",
            job.id,
            job.product_job_id,
        )
        return False
    finally:
        if owns_client:
            http.close()
