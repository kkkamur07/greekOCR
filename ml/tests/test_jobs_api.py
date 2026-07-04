"""ML job submit API tests."""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from ml.api.app import create_app
from ml.contracts.common import MLJobStatus, MLTask
from ml.infrastructure.job_repository import get_job_by_id

pytestmark = pytest.mark.integration


def test_submit_job_returns_ml_job_id():
    client = TestClient(create_app())
    product_job_id = uuid4()

    response = client.post(
        "/ml/v1/jobs",
        json={
            "task": "segment",
            "registry_model_id": "kraken-blla",
            "registry_tag": "stable",
            "product_job_id": str(product_job_id),
            "image_bytes": "cGFnZS1ieXRlcw==",
        },
    )

    assert response.status_code == 201
    ml_job_id = UUID(response.json()["ml_job_id"])
    job = get_job_by_id(ml_job_id)
    assert job is not None
    assert job.product_job_id == product_job_id
    assert job.task == MLTask.segment
    assert job.status == MLJobStatus.pending
    assert job.image_bytes == b"page-bytes"


def test_submit_job_rejects_unknown_registry_model():
    client = TestClient(create_app())

    response = client.post(
        "/ml/v1/jobs",
        json={
            "task": "segment",
            "registry_model_id": "missing-model",
            "product_job_id": str(uuid4()),
            "image_bytes": "YQ==",
        },
    )

    assert response.status_code == 400
    assert "unknown registry model" in response.json()["detail"]


def test_submit_job_rejects_task_mismatch():
    client = TestClient(create_app())

    response = client.post(
        "/ml/v1/jobs",
        json={
            "task": "transcribe",
            "registry_model_id": "kraken-blla",
            "product_job_id": str(uuid4()),
            "image_bytes": "YQ==",
        },
    )

    assert response.status_code == 400
    assert "does not match registry model task" in response.json()["detail"]
