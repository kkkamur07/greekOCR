"""Shared pytest fixtures — integration tests use real Postgres (kalamos)."""

import pytest
from fastapi.testclient import TestClient

from backend.core.app import create_app


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient against the real app (DB must be up: docker compose up db -d)."""
    return TestClient(create_app())
