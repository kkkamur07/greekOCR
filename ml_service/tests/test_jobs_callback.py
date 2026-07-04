"""Callback delivery and retry behavior."""

from __future__ import annotations

from uuid import uuid4

import httpx
from ml_service.contracts.common import MLJobStatus, MLTask
from ml_service.contracts.segment import SegmentLine, SegmentRunResponse
from ml_service.contracts.webhooks import ML_WEBHOOK_SECRET_HEADER
from ml_service.infrastructure.orm_models import MLJob
from ml_service.infrastructure.settings import MLSettings
from ml_service.jobs.callback import post_job_callback


def _sample_job() -> MLJob:
    return MLJob(
        id=uuid4(),
        product_job_id=uuid4(),
        task=MLTask.segment,
        registry_model_id="kraken-blla",
        registry_tag="stable",
        status=MLJobStatus.done,
        image_bytes=b"page",
        params={},
    )


def test_callback_retries_until_success(ml_settings: MLSettings):
    job = _sample_job()
    ml_settings.ml_callback_url = "http://callback.test/internal/ml/job-complete"
    ml_settings.ml_webhook_secret = "secret"
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(503)
        assert request.headers[ML_WEBHOOK_SECRET_HEADER] == "secret"
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        ok = post_job_callback(
            job,
            status=MLJobStatus.done,
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
            settings=ml_settings,
            client=client,
        )

    assert ok is True
    assert attempts["count"] == 3


def test_callback_logs_failure_after_max_retries(ml_settings: MLSettings, caplog):
    job = _sample_job()
    ml_settings.ml_callback_url = "http://callback.test/internal/ml/job-complete"
    ml_settings.ml_webhook_secret = "secret"

    transport = httpx.MockTransport(lambda _request: httpx.Response(500))
    with httpx.Client(transport=transport) as client:
        ok = post_job_callback(
            job,
            status=MLJobStatus.failed,
            error="boom",
            settings=ml_settings,
            client=client,
        )

    assert ok is False
    assert "exhausted retries" in caplog.text
