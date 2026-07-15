"""Shared pytest fixtures — light integration tests (Postgres + TestClient only).

ML-heavy tests under ``tests/nomicous/integration/ml/`` use a separate conftest that
boots the platform and inference services over real HTTP.
"""

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")
os.environ.setdefault("INFERENCE_URL", "http://localhost:8001")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:dev@localhost:5433/kalamos",
)
os.environ.setdefault(
    "SYNC_DATABASE_URL",
    "postgresql://postgres:dev@localhost:5433/kalamos",
)
os.environ.setdefault(
    "INFERENCE_DATABASE_URL",
    "postgresql://postgres:dev@localhost:5433/kalamos",
)
os.environ.setdefault("AUTH_RATE_LIMIT_REQUESTS", "1000")
os.environ.setdefault("ENABLE_TEST_JOB_ROUTES", "true")
os.environ.setdefault("JOB_WORKER_ENABLED", "true")
os.environ.setdefault("INFERENCE_WEBHOOK_SECRET", "test-inference-webhook-secret")
os.environ.setdefault("INFERENCE_SERVICE_SECRET", "test-inference-webhook-secret")
os.environ.setdefault(
    "MIGRATOR_DATABASE_URL",
    os.environ.get(
        "MIGRATOR_DATABASE_URL",
        "postgresql://postgres:dev@localhost:5433/kalamos",
    ),
)
os.environ.setdefault(
    "INFERENCE_REGISTRY_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../inference/registry.yaml")),
)

import infrastructure.models  # noqa: F401 — register all mappers
from backend.core.app import create_app
from backend.core.settings import (
    get_app_settings,
    get_auth_settings,
    get_infrastructure_settings,
    get_job_settings,
    get_ml_settings,
)
from backend.users.api.rate_limit import clear_auth_rate_limit_state

from infrastructure.db import Base, sync_engine

_truncate_engine = sync_engine
_TRUNCATE_ADVISORY_LOCK_ID = 73450123


def _ensure_job_status_cancelled() -> None:
    """Dev/test DBs created before cancelled existed need the enum value."""
    with _truncate_engine.begin() as connection:
        connection.execute(text("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'cancelled'"))


def _truncate_database() -> None:
    table_names = [
        sync_engine.dialect.identifier_preparer.quote(table.name)
        for table in reversed(Base.metadata.sorted_tables)
    ]
    for attempt in range(8):
        try:
            with _truncate_engine.begin() as connection:
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_id)"),
                    {"lock_id": _TRUNCATE_ADVISORY_LOCK_ID},
                )
                if table_names:
                    connection.execute(
                        text(f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE")
                    )
                connection.execute(text("TRUNCATE TABLE inference_jobs RESTART IDENTITY CASCADE"))
            return
        except OperationalError as exc:
            if "deadlock" not in str(exc).lower() or attempt == 7:
                raise
            time.sleep(0.1 * (attempt + 1))


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session TestClient — lifespan runs the platform job worker."""
    _ensure_job_status_cancelled()
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def isolated_platform_state(client: TestClient):
    from backend.jobs.infrastructure import worker as worker_module

    get_app_settings.cache_clear()
    get_auth_settings.cache_clear()
    get_infrastructure_settings.cache_clear()
    get_job_settings.cache_clear()
    get_ml_settings.cache_clear()
    clear_auth_rate_limit_state()

    worker_module._inference_client = None
    _truncate_database()

    yield
    clear_auth_rate_limit_state()
    worker_module._inference_client = None


@pytest.fixture
def unique_user() -> dict[str, str]:
    """Unique credentials dict — used with live register HTTP calls."""
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
