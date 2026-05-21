"""Shared pytest fixtures — integration tests use real Postgres (kalamos)."""

import pytest
from fastapi.testclient import TestClient

import infrastructure.models  # noqa: F401 — register all ORM mappers
from backend.core.app import create_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Single app + worker for the session — avoids overlapping lifespan workers."""
    with TestClient(create_app()) as test_client:
        yield test_client
