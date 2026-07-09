"""ML job submit API tests."""

from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from inference.contracts.common import InferenceJobStatus, InferenceTask
from inference.infrastructure.job_repository import get_job_by_id

pytestmark = pytest.mark.integration


# --- Job submit API (happy path) ---
# Tests POST creates a pending inference_jobs row. 


def test_submit_job_returns_inference_job_id(
    submit_inference_job: Callable[..., tuple[Response, UUID]],
):
    product_job_id = uuid4()

    response, _ = submit_inference_job(
        task=InferenceTask.segment,
        registry_model_id="greek-kraken-segment-v1",
        product_job_id=product_job_id,
        image_bytes=b"page-bytes",
    )

    assert response.status_code == 201
    inference_job_id = UUID(response.json()["inference_job_id"])
    job = get_job_by_id(inference_job_id)
    
    assert job is not None
    assert job.product_job_id == product_job_id
    assert job.task == InferenceTask.segment
    assert job.status == InferenceJobStatus.pending
    assert job.image_bytes == b"page-bytes"


# --- Job submit validation ---
# Tests registry and task mismatch errors. Does not test auth or callback delivery.


def test_submit_job_rejects_unknown_registry_model(
    submit_inference_job: Callable[..., tuple[Response, UUID]],
):
    response, _ = submit_inference_job(
        task=InferenceTask.segment,
        registry_model_id="missing-model",
        image_bytes=b"a",
    )

    assert response.status_code == 400
    assert "unknown registry model" in response.json()["detail"]


def test_submit_job_rejects_task_mismatch(
    submit_inference_job: Callable[..., tuple[Response, UUID]],
):
    response, _ = submit_inference_job(
        task=InferenceTask.transcribe,
        registry_model_id="greek-kraken-segment-v1",
        image_bytes=b"a",
        params={"lines": [{"line_index": 0}]},
    )

    assert response.status_code == 400
    assert "does not match registry model task" in response.json()["detail"]


def test_submit_job_requires_service_secret():
    from inference.api.app import create_app

    client = TestClient(create_app())
    response = client.post(
        "/inference/v1/jobs",
        json={
            "task": "segment",
            "registry_model_id": "greek-kraken-segment-v1",
            "registry_tag": "stable",
            "product_job_id": str(uuid4()),
            "image_bytes": "YQ==",
            "params": {},
        },
    )
    assert response.status_code == 401
