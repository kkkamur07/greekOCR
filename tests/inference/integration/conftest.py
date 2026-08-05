"""Pytest fixtures for inference service integration tests.

There is no database here. ADR 0003 collapsed the inference service's own job
queue into the platform's, so what remains is a stateless HTTP surface over the
registry and the model runner.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("INFERENCE_REGISTRY_PATH", "inference/registry.yaml")
os.environ.setdefault("INFERENCE_SERVICE_SECRET", "test-inference-service-secret")

from inference.api.app import create_app
from inference.contracts.webhooks import INFERENCE_SERVICE_SECRET_HEADER
from inference.settings import get_inference_settings


@pytest.fixture(autouse=True)
def isolated_inference_state():
    get_inference_settings.cache_clear()
    yield
    get_inference_settings.cache_clear()


@pytest.fixture
def inference_client() -> TestClient:
    settings = get_inference_settings()
    headers: dict[str, str] = {}
    if settings.inference_service_secret:
        headers[INFERENCE_SERVICE_SECRET_HEADER] = settings.inference_service_secret
    return TestClient(create_app(), headers=headers)
