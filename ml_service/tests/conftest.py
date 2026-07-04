"""Pytest fixtures for ML service tests (Postgres-backed integration)."""

from __future__ import annotations

import os
from base64 import b64encode
from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from ml_service.api.app import create_app
from ml_service.contracts.common import MLTask
from sqlalchemy import text

os.environ.setdefault("DATABASE_URL", "postgresql://postgres:dev@localhost:5433/kalamos")
os.environ.setdefault("ML_REGISTRY_PATH", "ml_service/registry.yaml")
os.environ.setdefault("ML_WEBHOOK_SECRET", "test-ml-webhook-secret")

from ml_service.infrastructure.db import engine, ensure_schema
from ml_service.infrastructure.settings import get_ml_settings


def _truncate_ml_jobs() -> None:
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE ml_jobs RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def isolated_ml_state(request: pytest.FixtureRequest):
    get_ml_settings.cache_clear()
    if request.node.get_closest_marker("integration") is None:
        yield
        get_ml_settings.cache_clear()
        return

    ensure_schema()
    _truncate_ml_jobs()
    yield
    get_ml_settings.cache_clear()
    _truncate_ml_jobs()


@pytest.fixture
def ml_settings(monkeypatch: pytest.MonkeyPatch):
    from ml_service.infrastructure.settings import MLSettings

    settings = MLSettings()
    get_ml_settings.cache_clear()
    monkeypatch.setattr("ml_service.infrastructure.settings.get_ml_settings", lambda: settings)
    return settings


@pytest.fixture
def ml_client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def submit_ml_job(
    ml_client: TestClient,
) -> Callable[..., tuple[Response, UUID]]:
    def _submit_ml_job(
        *,
        task: MLTask | str,
        registry_model_id: str,
        image_bytes: bytes,
        product_job_id: UUID | None = None,
        registry_tag: str = "stable",
        params: dict[str, object] | None = None,
    ) -> tuple[Response, UUID]:
        resolved_product_job_id = product_job_id or uuid4()
        response = ml_client.post(
            "/ml/v1/jobs",
            json={
                "task": task.value if isinstance(task, MLTask) else task,
                "registry_model_id": registry_model_id,
                "registry_tag": registry_tag,
                "product_job_id": str(resolved_product_job_id),
                "image_bytes": b64encode(image_bytes).decode(),
                "params": params or {},
            },
        )
        return response, resolved_product_job_id

    return _submit_ml_job
