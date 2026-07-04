"""ML job submit API tests."""

from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from httpx import Response
from ml_service.contracts.common import MLJobStatus, MLTask
from ml_service.infrastructure.job_repository import get_job_by_id

pytestmark = pytest.mark.integration


def test_submit_job_returns_ml_job_id(
    submit_ml_job: Callable[..., tuple[Response, UUID]],
):
    product_job_id = uuid4()

    response, _ = submit_ml_job(
        task=MLTask.segment,
        registry_model_id="kraken-blla",
        product_job_id=product_job_id,
        image_bytes=b"page-bytes",
    )

    assert response.status_code == 201
    ml_job_id = UUID(response.json()["ml_job_id"])
    job = get_job_by_id(ml_job_id)
    assert job is not None
    assert job.product_job_id == product_job_id
    assert job.task == MLTask.segment
    assert job.status == MLJobStatus.pending
    assert job.image_bytes == b"page-bytes"


def test_submit_job_rejects_unknown_registry_model(
    submit_ml_job: Callable[..., tuple[Response, UUID]],
):
    response, _ = submit_ml_job(
        task=MLTask.segment,
        registry_model_id="missing-model",
        image_bytes=b"a",
    )

    assert response.status_code == 400
    assert "unknown registry model" in response.json()["detail"]


def test_submit_job_rejects_task_mismatch(
    submit_ml_job: Callable[..., tuple[Response, UUID]],
):
    response, _ = submit_ml_job(
        task=MLTask.transcribe,
        registry_model_id="kraken-blla",
        image_bytes=b"a",
    )

    assert response.status_code == 400
    assert "does not match registry model task" in response.json()["detail"]
