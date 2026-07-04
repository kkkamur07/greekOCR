"""Shared pytest fixtures — integration tests use real Postgres (kalamos).

No DB mocking: ``unique_user`` only generates unique credentials; ``registered_user``
hits live ``POST /auth/register`` against kalamos.
"""

import json
import os
import socket
import threading
import time
import uuid

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import text

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")
os.environ.setdefault("AUTH_RATE_LIMIT_REQUESTS", "1000")
os.environ.setdefault("ENABLE_TEST_JOB_ROUTES", "true")
os.environ.setdefault("JOB_WORKER_ENABLED", "false")
os.environ.setdefault("ML_WEBHOOK_SECRET", "test-ml-webhook-secret")

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

_ACTIVE_ML_JOBS_MOCK: "MlJobsMock | None" = None
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
    if not table_names:
        return
    with sync_engine.begin() as connection:
        connection.execute(
            text(f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE")
        )


class MlJobsMock:
    def __init__(
        self,
        client: TestClient,
        *,
        auto_callback: bool = True,
        fail_submit: bool = False,
        callback_delay: float = 0.2,
    ) -> None:
        self.client = client
        self.auto_callback = auto_callback
        self.fail_submit = fail_submit
        self.callback_delay = callback_delay
        self.submitted: list[dict] = []
        self._errors: list[BaseException] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def install(self) -> None:
        from backend.jobs.infrastructure import worker as worker_module
        from backend.ml.infrastructure.ml_client import MlServiceClient

        get_ml_settings.cache_clear()
        worker_module._ml_client = MlServiceClient(
            base_url="http://ml.test",
            transport=httpx.MockTransport(self._handle_request),
        )

    def _deliver_callback(self, body, ml_job_id: uuid.UUID) -> None:
        def _post() -> None:
            from ml_service.contracts.common import MLJobStatus
            from ml_service.contracts.jobs import JobCallbackRequest
            from ml_service.contracts.transcribe import CharacterConfidence, TranscribeRunResponse
            from ml_service.architectures.mock import mock_segment
            from ml_service.jobs.callback import _wrap_job_output

            try:
                time.sleep(self.callback_delay)
                if body.task.value == "segment":
                    output = mock_segment(body.image_bytes)
                else:
                    line_index = int(body.params.get("line_index", 0))
                    text = f"mock transcription {line_index + 1}"
                    output = TranscribeRunResponse(
                        text=text,
                        confidence=0.99,
                        character_confidences=[
                            CharacterConfidence(char=char, confidence=0.99)
                            for char in text
                        ],
                    )
                callback = JobCallbackRequest(
                    ml_job_id=ml_job_id,
                    product_job_id=body.product_job_id,
                    task=body.task,
                    status=MLJobStatus.done,
                    output=_wrap_job_output(body.task, output),
                )
                self.client.post(
                    "/internal/ml/job-complete",
                    headers=_WEBHOOK_HEADERS,
                    json=callback.model_dump(mode="json"),
                )
            except BaseException as exc:  # pragma: no cover - surfaced by wait_for_callbacks
                with self._lock:
                    self._errors.append(exc)

        thread = threading.Thread(target=_post, daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def _handle_request(self, request: httpx.Request) -> httpx.Response:
        from ml_service.contracts.jobs import JobSubmitRequest

        if request.method == "POST" and request.url.path == "/ml/v1/jobs":
            if self.fail_submit:
                return httpx.Response(503, json={"detail": "ml unavailable"})
            body = JobSubmitRequest.model_validate(json.loads(request.content))
            ml_job_id = uuid.uuid4()
            self.submitted.append(body.model_dump(mode="json"))
            if self.auto_callback:
                self._deliver_callback(body, ml_job_id)
            return httpx.Response(201, json={"ml_job_id": str(ml_job_id)})

        return httpx.Response(404, json={"detail": "not found"})

    def wait_for_callbacks(self, *, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                threads = list(self._threads)
                self._threads.clear()
            if not threads:
                break
            for thread in threads:
                remaining = max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)
                if thread.is_alive():
                    raise AssertionError("ML callback thread did not finish before teardown")

        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
        if errors:
            raise AssertionError("ML callback thread failed") from errors[0]


def install_ml_jobs_mock(
    client: TestClient,
    *,
    auto_callback: bool = True,
    fail_submit: bool = False,
    callback_delay: float = 0.2,
) -> MlJobsMock:
    global _ACTIVE_ML_JOBS_MOCK
    mock = MlJobsMock(
        client,
        auto_callback=auto_callback,
        fail_submit=fail_submit,
        callback_delay=callback_delay,
    )
    mock.install()
    _ACTIVE_ML_JOBS_MOCK = mock
    return mock


def _wait_for_active_ml_jobs_mock() -> None:
    if _ACTIVE_ML_JOBS_MOCK is not None:
        _ACTIVE_ML_JOBS_MOCK.wait_for_callbacks()


@pytest.fixture(autouse=True)
def isolated_platform_state():
    _wait_for_active_ml_jobs_mock()
    get_app_settings.cache_clear()
    get_auth_settings.cache_clear()
    get_infrastructure_settings.cache_clear()
    get_job_settings.cache_clear()
    get_ml_settings.cache_clear()
    get_model_settings.cache_clear()
    clear_auth_rate_limit_state()
    _truncate_database()
    yield
    _wait_for_active_ml_jobs_mock()
    clear_auth_rate_limit_state()
    _truncate_database()

def _install_ml_jobs_mock(client: TestClient) -> None:
    """Submit segment/transcribe jobs to mocked ``POST /ml/v1/jobs`` and auto-callback."""
    install_ml_jobs_mock(client)


@pytest.fixture(scope="session", autouse=True)
def wire_mock_ml_jobs_service(client: TestClient) -> None:
    _install_ml_jobs_mock(client)


def restore_default_ml_jobs_mock(client: TestClient) -> MlJobsMock:
    """Re-install the session autouse ML jobs mock after per-test overrides."""
    return install_ml_jobs_mock(client)


@pytest.fixture(autouse=True)
def _restore_ml_jobs_mock_after_test(client: TestClient) -> None:
    yield
    _wait_for_active_ml_jobs_mock()
    restore_default_ml_jobs_mock(client)


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

    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("ML service test server did not start")

    yield url

    server.should_exit = True
    thread.join(timeout=5)
    worker_module._ml_client = None
    get_ml_settings.cache_clear()
    get_ml_service_settings.cache_clear()


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
