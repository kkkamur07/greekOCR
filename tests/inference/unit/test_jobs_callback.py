"""Callback delivery and retry behavior."""

from __future__ import annotations

from uuid import uuid4

import httpx
from inference.contracts.common import InferenceJobStatus, InferenceTask
from inference.contracts.segment import SegmentLine, SegmentRunResponse
from inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER
from inference.infrastructure.orm_models import InferenceJob
from inference.infrastructure.settings import InferenceSettings
from inference.jobs.callback import post_job_callback


def _sample_job() -> InferenceJob:
    return InferenceJob(
        id=uuid4(),
        product_job_id=uuid4(),
        task=InferenceTask.segment,
        registry_model_id="kraken-segment",
        registry_tag="stable",
        status=InferenceJobStatus.done,
        image_bytes=b"page",
        params={},
    )


# --- Callback HTTP delivery ---
# Tests retry and webhook header on outbound POST. Does not run a real inference job or platform server.


def test_callback_retries_until_success(inference_settings: InferenceSettings):
    job = _sample_job()
    inference_settings.inference_callback_url = (
        "http://callback.test/internal/inference/job-complete"
    )
    inference_settings.inference_webhook_secret = "secret"
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(503)
        assert request.headers[INFERENCE_WEBHOOK_SECRET_HEADER] == "secret"
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        ok = post_job_callback(
            job,
            status=InferenceJobStatus.done,
            output=SegmentRunResponse(
                lines=[
                    SegmentLine(
                        external_id="l1",
                        order=0,
                        baseline={"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
                        points=[[0, 0], [1, 0], [1, 1], [0, 1]],
                    )
                ]
            ),
            settings=inference_settings,
            client=client,
        )

    assert ok is True
    assert attempts["count"] == 3


# --- Callback exhaustion ---
# Tests logging when all retries fail. Does not test platform-side callback handling.


def test_callback_logs_failure_after_max_retries(inference_settings: InferenceSettings, caplog):
    job = _sample_job()
    inference_settings.inference_callback_url = (
        "http://callback.test/internal/inference/job-complete"
    )
    inference_settings.inference_webhook_secret = "secret"

    transport = httpx.MockTransport(lambda _request: httpx.Response(500))
    with httpx.Client(transport=transport) as client:
        ok = post_job_callback(
            job,
            status=InferenceJobStatus.failed,
            error="boom",
            settings=inference_settings,
            client=client,
        )

    assert ok is False
    assert "exhausted retries" in caplog.text
