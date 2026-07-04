"""End-to-end ML job queue: submit → worker runner → nomicous callback."""

from __future__ import annotations

import base64
import socket
import threading
import time
from io import BytesIO
from uuid import UUID, uuid4

import pytest
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.testclient import TestClient
from ml.architectures.mock import mock_segment
from ml.api.app import create_app
from ml.contracts.common import MLJobStatus, MLTask
from ml.contracts.jobs import JobCallbackRequest
from ml.contracts.webhooks import ML_WEBHOOK_SECRET_HEADER
from ml.infrastructure.job_repository import get_job_by_id
from ml.infrastructure.settings import MLSettings
from ml.jobs.worker import process_next_job
from PIL import Image

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _png(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def synthetic_worker_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _run_job(job):
        if job.task == MLTask.segment:
            return mock_segment(job.image_bytes)
        raise ValueError(f"unsupported test task: {job.task}")

    monkeypatch.setattr("ml.jobs.worker.run_job", _run_job)


@pytest.fixture
def nomicous_callback_server(ml_settings: MLSettings):
    """Minimal nomicous test app that accepts ML job completion callbacks."""
    received: list[JobCallbackRequest] = []
    secret = "integration-webhook-secret"
    ml_settings.ml_webhook_secret = secret

    app = FastAPI()

    @app.post("/internal/ml/job-complete", status_code=204)
    def job_complete(
        body: JobCallbackRequest,
        x_ml_webhook_secret: str = Header(alias=ML_WEBHOOK_SECRET_HEADER),
    ) -> Response:
        if x_ml_webhook_secret != secret:
            raise HTTPException(status_code=401, detail="invalid webhook secret")
        received.append(body)
        return Response(status_code=204)

    port = _free_port()
    ml_settings.ml_callback_url = f"http://127.0.0.1:{port}/internal/ml/job-complete"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("callback test server did not start")

    yield received

    server.should_exit = True
    thread.join(timeout=5)


def test_submit_worker_callback_flow(nomicous_callback_server: list[JobCallbackRequest]):
    product_job_id = uuid4()
    ml_client = TestClient(create_app())

    submit = ml_client.post(
        "/ml/v1/jobs",
        json={
            "task": "segment",
            "registry_model_id": "kraken-blla",
            "registry_tag": "stable",
            "product_job_id": str(product_job_id),
            "image_bytes": base64.b64encode(_png(10, 20)).decode(),
        },
    )
    assert submit.status_code == 201
    ml_job_id = UUID(submit.json()["ml_job_id"])

    assert process_next_job() is True
    assert process_next_job() is False

    job = get_job_by_id(ml_job_id)
    assert job is not None
    assert job.status == MLJobStatus.done
    assert job.output is not None
    assert job.output["blocks"][0]["external_id"] == "kraken-block-1"
    assert job.output["lines"][0]["source_metadata"] == {"adapter": "kraken_stub"}

    assert len(nomicous_callback_server) == 1
    callback = nomicous_callback_server[0]
    assert callback.ml_job_id == ml_job_id
    assert callback.product_job_id == product_job_id
    assert callback.task == MLTask.segment
    assert callback.status == MLJobStatus.done
    assert callback.output is not None
    assert callback.output.kind == "segment"
    assert callback.output.data.blocks[0].external_id == "kraken-block-1"
    assert callback.output.data.lines[0].source_metadata == {"adapter": "kraken_stub"}
    assert callback.error is None
