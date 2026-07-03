"""Shared pytest fixtures — integration tests use real Postgres (kalamos).

No DB mocking: ``unique_user`` only generates unique credentials; ``registered_user``
hits live ``POST /auth/register`` against kalamos.
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")
os.environ.setdefault("AUTH_RATE_LIMIT_REQUESTS", "1000")
os.environ.setdefault("ENABLE_TEST_JOB_ROUTES", "true")
os.environ.setdefault("ML_WEBHOOK_SECRET", "test-ml-webhook-secret")
os.environ.setdefault("ML_SERVICE_URL", "http://ml.test")

import infrastructure.models  # noqa: F401 — register all ORM mappers
from backend.core.app import create_app
from backend.core.settings import (
    get_app_settings,
    get_auth_settings,
    get_infrastructure_settings,
    get_job_settings,
    get_ml_settings,
    get_model_settings,
)
from backend.users.api.rate_limit import clear_auth_rate_limit_state
from infrastructure.db import Base, sync_engine


def _truncate_database() -> None:
    table_names = [
        sync_engine.dialect.identifier_preparer.quote(table.name)
        for table in reversed(Base.metadata.sorted_tables)
    ]
    if not table_names:
        return
    with sync_engine.begin() as connection:
        connection.execute(
            text(f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE")
        )


@pytest.fixture(autouse=True)
def isolated_platform_state():
    get_app_settings.cache_clear()
    get_auth_settings.cache_clear()
    get_infrastructure_settings.cache_clear()
    get_job_settings.cache_clear()
    get_ml_settings.cache_clear()
    get_model_settings.cache_clear()
    clear_auth_rate_limit_state()
    _truncate_database()
    yield
    clear_auth_rate_limit_state()
    _truncate_database()


@pytest.fixture(scope="session", autouse=True)
def wire_in_process_ml_service() -> None:
    """Route segment jobs to the in-process ML run handler."""
    import json

    import httpx

    from backend.jobs.infrastructure import worker as worker_module
    from backend.ml.infrastructure.ml_client import MlServiceClient
    from ml.api.run import run_ml
    from ml.contracts.run import MlRunRequest

    def _ml_run_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/ml/v1/run":
            body = MlRunRequest.model_validate(json.loads(request.content))
            output = run_ml(body)
            return httpx.Response(200, json=output.model_dump(mode="json"))
        return httpx.Response(404, json={"detail": "not found"})

    get_ml_settings.cache_clear()
    worker_module._ml_client = MlServiceClient(
        base_url="http://ml.test",
        transport=httpx.MockTransport(_ml_run_handler),
    )


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session TestClient — one asyncio loop; lifespan starts the job worker."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def unique_user() -> dict[str, str]:
    """Unique credentials dict only — not a mock; used with live register HTTP calls."""
    suffix = uuid.uuid4().hex[:8]
    return {
        "email": f"user-{suffix}@test.kalamos",
        "username": f"user_{suffix}",
        "password": "test-pass-123",
    }


@pytest.fixture
def registered_user(client: TestClient, unique_user: dict[str, str]) -> dict[str, str]:
    """Register a user and return credentials plus access_token."""
    response = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    assert response.status_code == 201
    data = unique_user.copy()
    data["access_token"] = response.json()["access_token"]
    return data


@pytest.fixture
def auth_headers(registered_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {registered_user['access_token']}"}


@pytest.fixture
def test_user(registered_user: dict[str, str]) -> dict[str, str]:
    """Seeded test user for integration tests (via register fixture)."""
    return registered_user


def _register_user(client: TestClient, suffix: str) -> dict[str, str]:
    creds = {
        "email": f"user-{suffix}@test.kalamos",
        "username": f"user_{suffix}",
        "password": "test-pass-123",
    }
    response = client.post(
        "/auth/register",
        json={
            "email": creds["email"],
            "username": creds["username"],
            "password": creds["password"],
        },
    )
    assert response.status_code == 201
    creds["access_token"] = response.json()["access_token"]
    return creds


@pytest.fixture
def owner_user(client: TestClient) -> dict[str, str]:
    return _register_user(client, f"owner-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def collaborator_user(client: TestClient) -> dict[str, str]:
    return _register_user(client, f"collab-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def outsider_user(client: TestClient) -> dict[str, str]:
    return _register_user(client, f"outsider-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def owner_headers(owner_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {owner_user['access_token']}"}


@pytest.fixture
def collaborator_headers(collaborator_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {collaborator_user['access_token']}"}


@pytest.fixture
def outsider_headers(outsider_user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {outsider_user['access_token']}"}


@pytest.fixture
def owner_project(client: TestClient, owner_headers: dict[str, str]) -> dict:
    """Project owned by ``owner_user``."""
    slug = f"proj-{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Test project"},
    )
    assert response.status_code == 201
    return response.json()
