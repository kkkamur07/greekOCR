"""Pytest fixtures for ML service tests (Postgres-backed integration)."""

from __future__ import annotations

import os

import pytest
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
