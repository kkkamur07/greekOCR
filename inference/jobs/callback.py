"""POST terminal inference job status to nomicous with retries."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from inference.contracts.common import InferenceJobStatus, InferenceTask
from inference.contracts.jobs import (
    JobCallbackRequest,
    SegmentJobOutput,
    TranscribeJobOutput,
)
from inference.contracts.segment import SegmentRunResponse
from inference.contracts.transcribe import TranscribeBatchRunResponse
from inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER
from inference.infrastructure.orm_models import InferenceJob
from inference.infrastructure.settings import InferenceSettings

logger = logging.getLogger(__name__)

CALLBACK_MAX_ATTEMPTS = 4
CALLBACK_RETRY_DELAY_SECONDS = 0.5


def _wrap_job_output(
    task: InferenceTask,
    output: SegmentRunResponse | TranscribeBatchRunResponse,
) -> SegmentJobOutput | TranscribeJobOutput:
    if task == InferenceTask.segment:
        if not isinstance(output, SegmentRunResponse):
            raise ValueError("segment callbacks require segment output")
        return SegmentJobOutput(kind="segment", data=output)
    if task == InferenceTask.transcribe:
        if not isinstance(output, TranscribeBatchRunResponse):
            raise ValueError("transcribe callbacks require batched output")
        return TranscribeJobOutput(kind="transcribe", data=output)
    raise ValueError(f"unsupported callback task: {task.value}")


def _build_callback_payload(
    job: InferenceJob,
    *,
    status: InferenceJobStatus,
    output: SegmentRunResponse | TranscribeBatchRunResponse | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    wrapped_output = (
        _wrap_job_output(InferenceTask(job.task), output) if output is not None else None
    )
    callback = JobCallbackRequest(
        inference_job_id=job.id,
        product_job_id=job.product_job_id,
        task=InferenceTask(job.task),
        status=status,
        output=wrapped_output,
        error=error,
    )
    return callback.model_dump(mode="json")


def post_job_callback(
    job: InferenceJob,
    *,
    status: InferenceJobStatus,
    output: SegmentRunResponse | TranscribeBatchRunResponse | None = None,
    error: str | None = None,
    settings: InferenceSettings | None = None,
    client: httpx.Client | None = None,
) -> bool:
    from inference.infrastructure.settings import get_inference_settings

    resolved_settings = settings or get_inference_settings()
    if not resolved_settings.inference_callback_url:
        logger.error(
            "INFERENCE_CALLBACK_URL is not configured; skipping callback for inference_job_id=%s",
            job.id,
        )
        return False
    if not resolved_settings.inference_webhook_secret:
        logger.error(
            "INFERENCE_WEBHOOK_SECRET is not configured; skipping callback for inference_job_id=%s",
            job.id,
        )
        return False

    payload = _build_callback_payload(job, status=status, output=output, error=error)
    headers = {INFERENCE_WEBHOOK_SECRET_HEADER: resolved_settings.inference_webhook_secret}
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)

    try:
        for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
            try:
                response = http.post(
                    resolved_settings.inference_callback_url,
                    json=payload,
                    headers=headers,
                )
                if response.status_code in (200, 204):
                    logger.info(
                        "callback delivered for inference_job_id=%s product_job_id=%s status=%s",
                        job.id,
                        job.product_job_id,
                        status.value,
                    )
                    return True
                logger.warning(
                    "callback attempt %s/%s failed for inference_job_id=%s: HTTP %s",
                    attempt,
                    CALLBACK_MAX_ATTEMPTS,
                    job.id,
                    response.status_code,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "callback attempt %s/%s failed for inference_job_id=%s: %s",
                    attempt,
                    CALLBACK_MAX_ATTEMPTS,
                    job.id,
                    exc,
                )
            if attempt < CALLBACK_MAX_ATTEMPTS:
                time.sleep(CALLBACK_RETRY_DELAY_SECONDS)
        logger.error(
            "callback exhausted retries for inference_job_id=%s product_job_id=%s",
            job.id,
            job.product_job_id,
        )
        return False
    finally:
        if owns_client:
            http.close()
