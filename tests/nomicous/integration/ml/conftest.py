"""ML-heavy integration fixtures — real HTTP for platform callbacks and inference API."""

from __future__ import annotations

import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn

pytestmark = pytest.mark.ml

_INFERENCE_WORKER_STOP = threading.Event()
_INFERENCE_WORKER_PAUSE = threading.Event()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_uvicorn(server: uvicorn.Server, thread: threading.Thread, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("test server did not start")


def _ml_worker_loop() -> None:
    from inference.jobs.worker import process_next_job

    while not _INFERENCE_WORKER_STOP.is_set():
        if _INFERENCE_WORKER_PAUSE.is_set():
            time.sleep(0.05)
            continue
        try:
            if not process_next_job():
                time.sleep(0.05)
        except Exception:
            time.sleep(0.1)


@pytest.fixture(scope="session")
def real_platform_url() -> str:
    """Run the platform app over HTTP so ML callbacks can reach it."""
    from backend.core.app import create_app

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_uvicorn(server, thread)

    yield url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def real_inference_url(real_platform_url: str) -> str:
    """Run inference API + background worker with callback URL wired to the platform."""
    from backend.core.settings import get_ml_settings
    from backend.jobs.infrastructure import worker as worker_module
    from inference.api.app import create_app as create_inference_app
    from inference.infrastructure.settings import get_inference_settings

    os.environ["INFERENCE_CALLBACK_URL"] = (
        f"{real_platform_url}/internal/inference/job-complete"
    )

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    os.environ["INFERENCE_URL"] = url

    get_ml_settings.cache_clear()
    get_inference_settings.cache_clear()
    worker_module._inference_client = None

    config = uvicorn.Config(create_inference_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_uvicorn(server, thread)

    worker_thread = threading.Thread(target=_ml_worker_loop, daemon=True)
    worker_thread.start()

    yield url

    _INFERENCE_WORKER_STOP.set()
    worker_thread.join(timeout=5)
    server.should_exit = True
    thread.join(timeout=5)
    worker_module._inference_client = None
    get_ml_settings.cache_clear()
    get_inference_settings.cache_clear()


@pytest.fixture(scope="session")
def client(real_inference_url: str, real_platform_url: str):
    """HTTP client against the live platform — overrides parent TestClient to avoid dual event loops."""
    with httpx.Client(base_url=real_platform_url, timeout=60.0) as http_client:
        yield http_client


@pytest.fixture
def platform_http_client(client):
    """Alias for tests that name this fixture explicitly."""
    return client


@pytest.fixture
def paused_ml_worker():
    """Pause the background ML worker thread (for submit-without-callback tests)."""
    _INFERENCE_WORKER_PAUSE.set()
    yield
    _INFERENCE_WORKER_PAUSE.clear()


@pytest.fixture
def broken_inference(client):
    """Point the platform worker at an unreachable ML service URL."""
    from backend.core.settings import get_ml_settings
    from backend.jobs.infrastructure import worker as worker_module
    from backend.ml.infrastructure.ml_client import InferenceClient

    worker_module._inference_client = InferenceClient(base_url="http://127.0.0.1:1", timeout=0.2)
    yield
    worker_module._inference_client = None
    get_ml_settings.cache_clear()
