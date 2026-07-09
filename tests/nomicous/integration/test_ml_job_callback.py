"""Inference job completion callback: webhook auth, status transition, idempotency."""

from __future__ import annotations

import uuid

from backend.core.settings import get_inference_settings
from backend.document.infrastructure.orm_models import Document, DocumentPart, Line, LineGeometryKind
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.project.infrastructure.orm_models import Project
from fastapi.testclient import TestClient
from inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER

from infrastructure.db import sync_system_session

CALLBACK_URL = "/internal/inference/job-complete"
WEBHOOK_HEADERS = {INFERENCE_WEBHOOK_SECRET_HEADER: "test-inference-webhook-secret"}


def _segment_done_payload(
    *,
    product_job_id: uuid.UUID,
    inference_job_id: uuid.UUID,
) -> dict:
    return {
        "inference_job_id": str(inference_job_id),
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
    inference_job_id: uuid.UUID,
    line_id: uuid.UUID,
) -> dict:
    return {
        "inference_job_id": str(inference_job_id),
        "product_job_id": str(product_job_id),
        "task": "transcribe",
        "status": "done",
        "output": {
            "kind": "transcribe",
            "data": {
                "lines": [
                    {
                        "line_id": str(line_id),
                        "line_index": 0,
                        "output": {
                            "text": "Αβ",
                            "confidence": 0.91,
                            "character_confidences": [
                                {"char": "Α", "confidence": 0.93},
                                {"char": "β", "confidence": 0.89},
                            ],
                        },
                    }
                ]
            },
        },
    }


def _failed_payload(
    *,
    product_job_id: uuid.UUID,
    inference_job_id: uuid.UUID,
) -> dict:
    return {
        "inference_job_id": str(inference_job_id),
        "product_job_id": str(product_job_id),
        "task": "segment",
        "status": "failed",
        "error": "weights not found in cache",
    }


def _seed_waiting_job(
    *,
    product_job_id: uuid.UUID | None = None,
    inference_job_id: uuid.UUID | None = None,
    job_type: JobType = JobType.segment,
) -> tuple[uuid.UUID, uuid.UUID]:
    product_job_id = product_job_id or uuid.uuid4()
    inference_job_id = inference_job_id or uuid.uuid4()
    document_id = None
    part_id = None
    with sync_system_session() as session:
        if job_type == JobType.segment:
            project_id = uuid.uuid4()
            document_id = uuid.uuid4()
            part_id = uuid.uuid4()
            session.add(Project(id=project_id, name="Callback test", slug=f"callback-{uuid.uuid4().hex}"))
            session.flush()
            session.add(
                Document(
                    id=document_id,
                    project_id=project_id,
                    name="Test document",
                )
            )
            session.flush()
            session.add(
                DocumentPart(
                    id=part_id,
                    document_id=document_id,
                    image_key="test/page.png",
                )
            )
            session.flush()
        session.add(
            Job(
                id=product_job_id,
                type=job_type,
                status=JobStatus.waiting,
                payload={},
                inference_job_id=inference_job_id,
                document_id=document_id,
                document_part_id=part_id,
            )
        )
        session.commit()
    return product_job_id, inference_job_id


def _seed_transcribe_waiting_job(
    *,
    product_job_id: uuid.UUID | None = None,
    inference_job_id: uuid.UUID | None = None,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    product_job_id = product_job_id or uuid.uuid4()
    inference_job_id = inference_job_id or uuid.uuid4()
    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    part_id = uuid.uuid4()
    line_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(Project(id=project_id, name="Callback test", slug=f"callback-{uuid.uuid4().hex}"))
        session.flush()
        session.add(
            Document(
                id=document_id,
                project_id=project_id,
                name="Test document",
            )
        )
        session.flush()
        session.add(
            DocumentPart(
                id=part_id,
                document_id=document_id,
                image_key="test/page.png",
            )
        )
        session.flush()
        session.add(
            Line(
                id=line_id,
                part_id=part_id,
                baseline={"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                points=[[1, 1], [2, 1], [2, 2], [1, 2]],
                kind=LineGeometryKind.polygon,
            )
        )
        session.flush()
        session.add(
            Job(
                id=product_job_id,
                type=JobType.transcribe,
                status=JobStatus.waiting,
                payload={},
                inference_job_id=inference_job_id,
                document_id=document_id,
                document_part_id=part_id,
            )
        )
        session.commit()
    return product_job_id, inference_job_id, line_id


def _get_job(job_id: uuid.UUID) -> Job:
    with sync_system_session() as session:
        job = session.get(Job, job_id)
        assert job is not None
        session.expunge(job)
        return job


# --- Webhook authentication ---
# Tests secret header validation on the callback endpoint. Does not merge inference output into documents.


def test_callback_missing_secret_returns_401(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        json=_segment_done_payload(product_job_id=product_job_id, inference_job_id=inference_job_id),
    )
    assert response.status_code == 401


def test_callback_wrong_secret_returns_403(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers={INFERENCE_WEBHOOK_SECRET_HEADER: "wrong-secret"},
        json=_segment_done_payload(product_job_id=product_job_id, inference_job_id=inference_job_id),
    )
    assert response.status_code == 403


def test_callback_unconfigured_secret_returns_503(client: TestClient, monkeypatch):
    monkeypatch.delenv("INFERENCE_WEBHOOK_SECRET", raising=False)
    get_inference_settings.cache_clear()
    product_job_id, inference_job_id = _seed_waiting_job()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, inference_job_id=inference_job_id),
    )

    assert response.status_code == 503
    get_inference_settings.cache_clear()


# --- Successful callbacks ---
# Tests waiting jobs transition to done with merged results. Does not run real inference.


def test_callback_success_marks_job_done(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, inference_job_id=inference_job_id),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.inference_job_id == inference_job_id
    assert job.error is None
    assert job.result is not None
    assert job.result["blocks_count"] == 0
    assert job.result["lines_count"] == 1
    assert job.result["added_lines"] == 1
    assert job.result["pruned_lines"] == 0
    assert job.result["preserved_manual_lines"] == 0
    assert job.completed_at is not None


def test_callback_transcribe_success_marks_job_done(client: TestClient):
    product_job_id, inference_job_id, line_id = _seed_transcribe_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_transcribe_done_payload(
            product_job_id=product_job_id,
            inference_job_id=inference_job_id,
            line_id=line_id,
        ),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.inference_job_id == inference_job_id
    assert job.result is not None
    assert job.result["transcription_id"]
    assert job.result["lines"][0]["line_id"] == str(line_id)
    assert job.result["lines"][0]["text"] == "Αβ"
    assert job.result["lines"][0]["confidence"] == 0.91


# --- Failed callbacks ---
# Tests error persistence on terminal failure. Does not retry delivery from the platform side.


def test_callback_failure_marks_job_failed(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_failed_payload(product_job_id=product_job_id, inference_job_id=inference_job_id),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.failed
    assert job.error == "weights not found in cache"
    assert job.completed_at is not None


# --- Idempotency and validation ---
# Tests duplicate callbacks and mismatched job metadata. Does not enqueue new inference work.


def test_callback_on_terminal_job_is_idempotent(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    payload = _segment_done_payload(product_job_id=product_job_id, inference_job_id=inference_job_id)

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
    product_job_id, inference_job_id = _seed_waiting_job(job_type=JobType.transcribe)
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, inference_job_id=inference_job_id),
    )
    assert response.status_code == 409


def test_callback_ml_job_mismatch_returns_409(client: TestClient):
    product_job_id, seeded_inference_job_id = _seed_waiting_job()
    callback_inference_job_id = uuid.uuid4()
    assert callback_inference_job_id != seeded_inference_job_id

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id,
            inference_job_id=callback_inference_job_id,
        ),
    )
    assert response.status_code == 409


def test_callback_unknown_job_returns_404(client: TestClient):
    product_job_id = uuid.uuid4()
    inference_job_id = uuid.uuid4()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(product_job_id=product_job_id, inference_job_id=inference_job_id),
    )
    assert response.status_code == 404
