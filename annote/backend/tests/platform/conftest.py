"""Shared pytest fixtures — integration tests use real Postgres (kalamos).

Platform tests use the real ML API and worker (no HTTP mocks). The in-process ML
service submits jobs to Postgres; a background thread drains ``ml_jobs`` and posts
callbacks into the platform TestClient.
"""

import os
import socket
import threading
import time
import uuid

import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")
os.environ.setdefault("AUTH_RATE_LIMIT_REQUESTS", "1000")
os.environ.setdefault("ENABLE_TEST_JOB_ROUTES", "true")
os.environ.setdefault("JOB_WORKER_ENABLED", "true")
os.environ.setdefault("ML_WEBHOOK_SECRET", "test-ml-webhook-secret")
os.environ.setdefault(
    "ML_REGISTRY_PATH",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../ml_service/registry.yaml")
    ),
)

import infrastructure.models  # noqa: F401 — register all mappers
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

_ML_WORKER_STOP = threading.Event()
_ML_WORKER_PAUSE = threading.Event()
_WEBHOOK_HEADERS = {"X-ML-Webhook-Secret": os.environ["ML_WEBHOOK_SECRET"]}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _truncate_database() -> None:
    table_names = [
        sync_engine.dialect.identifier_preparer.quote(table.name)
        for table in reversed(Base.metadata.sorted_tables)
    ]
    with sync_engine.begin() as connection:
        if table_names:
            connection.execute(
                text(f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE")
            )
        connection.execute(text("TRUNCATE TABLE ml_jobs RESTART IDENTITY CASCADE"))


def _ml_worker_loop() -> None:
    from ml_service.jobs.worker import process_next_job

    while not _ML_WORKER_STOP.is_set():
        if _ML_WORKER_PAUSE.is_set():
            time.sleep(0.05)
            continue
        try:
            if not process_next_job():
                time.sleep(0.05)
        except Exception:
            time.sleep(0.1)


def _wire_platform_callbacks(client: TestClient) -> None:
    """Deliver ML job callbacks into the platform app under test."""
    from ml_service.jobs import callback as callback_module

    callback_module._platform_test_client = client  # type: ignore[attr-defined]

    def post_to_platform(
        job,
        *,
        status,
        output=None,
        error=None,
        settings=None,
        client=None,
    ) -> bool:
        payload = callback_module._build_callback_payload(
            job,
            status=status,
            output=output,
            error=error,
        )
        test_client = callback_module._platform_test_client  # type: ignore[attr-defined]
        response = test_client.post(
            "/internal/ml/job-complete",
            headers=_WEBHOOK_HEADERS,
            json=payload,
        )
        return response.status_code in (200, 204)

    callback_module.post_job_callback = post_to_platform


@pytest.fixture(scope="session", autouse=True)
def real_ml_service_url() -> str:
    """Run the actual ML FastAPI service for platform integration tests."""
    from backend.jobs.infrastructure import worker as worker_module
    from ml_service.api.app import create_app as create_ml_app
    from ml_service.infrastructure.settings import get_ml_settings as get_ml_service_settings

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    os.environ["ML_SERVICE_URL"] = url
    get_ml_settings.cache_clear()
    get_ml_service_settings.cache_clear()
    worker_module._ml_client = None

    config = uvicorn.Config(create_ml_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("ML service test server did not start")

    worker_thread = threading.Thread(target=_ml_worker_loop, daemon=True)
    worker_thread.start()

    yield url

    _ML_WORKER_STOP.set()
    worker_thread.join(timeout=5)
    server.should_exit = True
    thread.join(timeout=5)
    worker_module._ml_client = None
    get_ml_settings.cache_clear()
    get_ml_service_settings.cache_clear()


@pytest.fixture(scope="session")
def client(real_ml_service_url: str) -> TestClient:
    """Session TestClient — lifespan runs the platform job worker."""
    with TestClient(create_app()) as test_client:
        _wire_platform_callbacks(test_client)
        yield test_client


@pytest.fixture(autouse=True)
def isolated_platform_state(client: TestClient):
    from backend.jobs.infrastructure import worker as worker_module

    get_app_settings.cache_clear()
    get_auth_settings.cache_clear()
    get_infrastructure_settings.cache_clear()
    get_job_settings.cache_clear()
    get_ml_settings.cache_clear()
    get_model_settings.cache_clear()
    clear_auth_rate_limit_state()
    worker_module._ml_client = None
    _truncate_database()
    yield
    clear_auth_rate_limit_state()
    worker_module._ml_client = None
    _truncate_database()


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


@pytest.fixture
def paused_ml_worker():
    """Pause the background ML worker thread (for submit-without-callback tests)."""
    _ML_WORKER_PAUSE.set()
    yield
    _ML_WORKER_PAUSE.clear()


@pytest.fixture
def broken_ml_service(client: TestClient):
    """Point the platform worker at an unreachable ML service URL."""
    from backend.jobs.infrastructure import worker as worker_module
    from backend.ml.infrastructure.ml_client import MlServiceClient

    worker_module._ml_client = MlServiceClient(base_url="http://127.0.0.1:1", timeout=0.2)
    yield
    worker_module._ml_client = None
    get_ml_settings.cache_clear()
