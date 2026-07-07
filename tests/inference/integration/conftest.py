"""Pytest fixtures for ML service tests (Postgres-backed integration)."""

from __future__ import annotations

import os
from base64 import b64encode
from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from inference.api.app import create_app
from inference.contracts.common import InferenceTask
from sqlalchemy import text

os.environ.setdefault("DATABASE_URL", "postgresql://postgres:dev@localhost:5433/kalamos")
os.environ.setdefault("INFERENCE_REGISTRY_PATH", "inference/registry.yaml")
os.environ.setdefault("INFERENCE_WEBHOOK_SECRET", "test-inference-webhook-secret")

from inference.infrastructure.db import Base, engine
from inference.infrastructure.settings import get_inference_settings


def ensure_schema() -> None:
    from inference.infrastructure.orm_models import InferenceJob  # noqa: F401

    Base.metadata.create_all(bind=engine)


def _truncate_inference_jobs() -> None:
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE inference_jobs RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def isolated_inference_state(request: pytest.FixtureRequest):
    get_inference_settings.cache_clear()
    if request.node.get_closest_marker("integration") is None:
        yield
        get_inference_settings.cache_clear()
        return

    ensure_schema()
    _truncate_inference_jobs()
    yield
    get_inference_settings.cache_clear()
    _truncate_inference_jobs()


@pytest.fixture
def inference_client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def submit_inference_job(
    inference_client: TestClient,
) -> Callable[..., tuple[Response, UUID]]:
    def _submit_inference_job(
        *,
        task: InferenceTask | str,
        registry_model_id: str,
        image_bytes: bytes,
        product_job_id: UUID | None = None,
        registry_tag: str = "stable",
        params: dict[str, object] | None = None,
    ) -> tuple[Response, UUID]:
        resolved_product_job_id = product_job_id or uuid4()
        response = inference_client.post(
            "/inference/v1/jobs",
            json={
                "task": task.value if isinstance(task, InferenceTask) else task,
                "registry_model_id": registry_model_id,
                "registry_tag": registry_tag,
                "product_job_id": str(resolved_product_job_id),
                "image_bytes": b64encode(image_bytes).decode(),
                "params": params or {},
            },
        )
        return response, resolved_product_job_id

    return _submit_inference_job
