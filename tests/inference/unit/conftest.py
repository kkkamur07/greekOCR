"""Pytest fixtures for ML service unit tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from inference.api.app import create_app


@pytest.fixture
def inference_settings(monkeypatch: pytest.MonkeyPatch):
    from inference.infrastructure.settings import InferenceSettings, get_inference_settings

    settings = InferenceSettings()
    get_inference_settings.cache_clear()
    monkeypatch.setattr(
        "inference.infrastructure.settings.get_inference_settings", lambda: settings
    )
    return settings


@pytest.fixture
def inference_client() -> TestClient:
    return TestClient(create_app())
