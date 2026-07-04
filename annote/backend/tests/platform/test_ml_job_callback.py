"""ML job completion callback — webhook auth, status transition, idempotency."""

from __future__ import annotations

import uuid

from backend.core.settings import get_ml_settings
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from fastapi.testclient import TestClient
from ml_service.contracts.webhooks import ML_WEBHOOK_SECRET_HEADER

from infrastructure.db import SyncSessionLocal

CALLBACK_URL = "/internal/ml/job-complete"
WEBHOOK_HEADERS = {ML_WEBHOOK_SECRET_HEADER: "test-ml-webhook-secret"}


def _segment_done_payload(
    *,
    product_job_id: uuid.UUID,
    ml_job_id: uuid.UUID,
) -> dict:
    return {
        "ml_job_id": str(ml_job_id),
        "product_job_id": str(product_job_id),
        "task": "segment",
        "status": "done",
        "output": {
            "kind": "segment",
            "data": {
                "lines": [
                    {
                        "external_id": "l1",
                        "order": 0,
                        "baseline": {"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                        "points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                    }
                ]
            },
        },
    }


def _transcribe_done_payload(
    *,
    product_job_id: uuid.UUID,
    ml_job_id: uuid.UUID,
) -> dict:
    return {
        "ml_job_id": str(ml_job_id),
        "product_job_id": str(product_job_id),
        "task": "transcribe",
        "status": "done",
        "output": {
            "kind": "transcribe",
            "data": {
                "text": "Αβ",
                "confidence": 0.91,
                "character_confidences": [
                    {"char": "Α", "confidence": 0.93},
                    {"char": "β", "confidence": 0.89},
                ],
            },
        },
    }


def _failed_payload(
    *,
    product_job_id: uuid.UUID,
    ml_job_id: uuid.UUID,
) -> dict:
    return {
        "ml_job_id": str(ml_job_id),
        "product_job_id": str(product_job_id),
        "task": "segment",
        "status": "failed",
        "error": "weights not found in cache",
    }


def _seed_waiting_job(
    *,
    product_job_id: uuid.UUID | None = None,
    ml_job_id: uuid.UUID | None = None,
    job_type: JobType = JobType.segment,
) -> tuple[uuid.UUID, uuid.UUID]:
    product_job_id = product_job_id or uuid.uuid4()
    ml_job_id = ml_job_id or uuid.uuid4()
    with SyncSessionLocal() as session:
        session.add(
            Job(
                id=product_job_id,
                type=job_type,
                status=JobStatus.waiting,
                payload={},
                ml_job_id=ml_job_id,
            )
        )
        session.commit()
    return product_job_id, ml_job_id


def _get_job(job_id: uuid.UUID) -> Job:
    with SyncSessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        session.expunge(job)
        return job


def test_callback_missing_secret_returns_401(client: TestClient):
    product_job_id, ml_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        json=_segment_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )
    assert response.status_code == 401


def test_callback_wrong_secret_returns_403(client: TestClient):
    product_job_id, ml_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers={ML_WEBHOOK_SECRET_HEADER: "wrong-secret"},
        json=_segment_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )
    assert response.status_code == 403


def test_callback_unconfigured_secret_returns_503(client: TestClient, monkeypatch):
    monkeypatch.delenv("ML_WEBHOOK_SECRET", raising=False)
    get_ml_settings.cache_clear()
    product_job_id, ml_job_id = _seed_waiting_job()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )

    assert response.status_code == 503
    get_ml_settings.cache_clear()


def test_callback_success_marks_job_done(client: TestClient):
    product_job_id, ml_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.ml_job_id == ml_job_id
    assert job.error is None
    assert job.result is not None
    assert job.result["ml_job_id"] == str(ml_job_id)
    assert job.result["task"] == "segment"
    assert job.result["output"]["kind"] == "segment"
    assert job.result["output"]["data"]["lines"][0]["external_id"] == "l1"
    assert job.completed_at is not None


def test_callback_transcribe_success_marks_job_done(client: TestClient):
    product_job_id, ml_job_id = _seed_waiting_job(job_type=JobType.transcribe)
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_transcribe_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.ml_job_id == ml_job_id
    assert job.result is not None
    assert job.result["task"] == "transcribe"
    assert job.result["output"]["kind"] == "transcribe"
    assert job.result["output"]["data"]["text"] == "Αβ"
    assert job.result["output"]["data"]["character_confidences"][0]["char"] == "Α"


def test_callback_failure_marks_job_failed(client: TestClient):
    product_job_id, ml_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_failed_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.failed
    assert job.error == "weights not found in cache"
    assert job.completed_at is not None


def test_callback_on_terminal_job_is_idempotent(client: TestClient):
    product_job_id, ml_job_id = _seed_waiting_job()
    payload = _segment_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id)

    first = client.post(CALLBACK_URL, headers=WEBHOOK_HEADERS, json=payload)
    assert first.status_code == 204
    after_first = _get_job(product_job_id)

    second = client.post(CALLBACK_URL, headers=WEBHOOK_HEADERS, json=payload)
    assert second.status_code == 204
    after_second = _get_job(product_job_id)

    assert after_second.status == JobStatus.done
    assert after_second.completed_at == after_first.completed_at
    assert after_second.result == after_first.result
    assert after_second.error == after_first.error


def test_callback_task_mismatch_returns_409(client: TestClient):
    product_job_id, ml_job_id = _seed_waiting_job(job_type=JobType.transcribe)
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )
    assert response.status_code == 409


def test_callback_ml_job_mismatch_returns_409(client: TestClient):
    product_job_id, seeded_ml_job_id = _seed_waiting_job()
    callback_ml_job_id = uuid.uuid4()
    assert callback_ml_job_id != seeded_ml_job_id

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id,
            ml_job_id=callback_ml_job_id,
        ),
    )
    assert response.status_code == 409


def test_callback_unknown_job_returns_404(client: TestClient):
    product_job_id = uuid.uuid4()
    ml_job_id = uuid.uuid4()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, ml_job_id=ml_job_id),
    )
    assert response.status_code == 404
