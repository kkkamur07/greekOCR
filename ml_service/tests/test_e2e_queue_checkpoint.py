"""Checkpoint-backed ML queue E2E: segment -> queue -> transcribe -> queue -> callbacks."""

from __future__ import annotations

import socket
import threading
import time
from base64 import b64encode
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.testclient import TestClient
from ml_service.api.app import create_app
from ml_service.contracts.common import MLJobStatus, MLTask
from ml_service.contracts.jobs import JobCallbackRequest
from ml_service.contracts.webhooks import ML_WEBHOOK_SECRET_HEADER
from ml_service.infrastructure.job_repository import get_job_by_id
from ml_service.infrastructure.orm_models import MLJob
from ml_service.infrastructure.settings import MLSettings
from ml_service.jobs.worker import process_next_job

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENT_IMAGE_PATH = (
    REPO_ROOT
    / "annote/data/manuscripts/pages/Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_6.jpeg"
)
TRANSCRIBE_IMAGE_PATH = (
    REPO_ROOT
    / "annote/data/manuscripts/export/Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_6_1.jpg"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _read_image_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"test image not found: {path}")
    return path.read_bytes()


@pytest.fixture
def checkpoint_callback_server(ml_settings: MLSettings):
    received: list[JobCallbackRequest] = []
    secret = "checkpoint-e2e-webhook-secret"
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


def _submit_job(
    client: TestClient,
    *,
    task: MLTask,
    registry_model_id: str,
    image_bytes: bytes,
) -> UUID:
    response = client.post(
        "/ml/v1/jobs",
        json={
            "task": task.value,
            "registry_model_id": registry_model_id,
            "registry_tag": "stable",
            "product_job_id": str(uuid4()),
            "image_bytes": b64encode(image_bytes).decode(),
            "params": {"line_index": 0} if task == MLTask.transcribe else {},
        },
    )
    response.raise_for_status()
    return UUID(response.json()["ml_job_id"])


def _process_and_assert_done(ml_job_id: UUID, *, expected_task: MLTask) -> MLJob:
    assert process_next_job() is True
    job = get_job_by_id(ml_job_id)
    assert job is not None
    assert job.status == MLJobStatus.done, job.error
    assert job.task == expected_task
    assert job.output is not None
    return job


def test_checkpoint_queue_runs_segment_then_transcribe_and_posts_outputs(
    checkpoint_callback_server: list[JobCallbackRequest],
):
    client = TestClient(create_app())

    segment_job_id = _submit_job(
        client,
        task=MLTask.segment,
        registry_model_id="kraken-blla",
        image_bytes=_read_image_bytes(SEGMENT_IMAGE_PATH),
    )
    segment_job = _process_and_assert_done(segment_job_id, expected_task=MLTask.segment)
    assert "lines" in segment_job.output
    assert segment_job.output["lines"]

    transcribe_job_id = _submit_job(
        client,
        task=MLTask.transcribe,
        registry_model_id="syriac-calamariv1",
        image_bytes=_read_image_bytes(TRANSCRIBE_IMAGE_PATH),
    )
    transcribe_job = _process_and_assert_done(
        transcribe_job_id,
        expected_task=MLTask.transcribe,
    )
    assert "text" in transcribe_job.output
    assert isinstance(transcribe_job.output.get("text"), str)

    assert process_next_job() is False
    assert [callback.task for callback in checkpoint_callback_server] == [
        MLTask.segment,
        MLTask.transcribe,
    ]
    assert checkpoint_callback_server[0].ml_job_id == segment_job_id
    assert checkpoint_callback_server[0].output is not None
    assert checkpoint_callback_server[0].output.kind == "segment"
    assert checkpoint_callback_server[1].ml_job_id == transcribe_job_id
    assert checkpoint_callback_server[1].output is not None
    assert checkpoint_callback_server[1].output.kind == "transcribe"
