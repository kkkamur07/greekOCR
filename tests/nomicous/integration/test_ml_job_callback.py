"""Inference job completion callback: webhook auth, status transition, idempotency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import uuid
from datetime import UTC, datetime

import pytest

from backend.core.settings import get_inference_settings
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    LineGeometryKind,
    Transcription,
)
from backend.jobs.application import job_callback_service
from backend.jobs.application.job_callback_service import INFERENCE_FAILURE_ERROR
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.project.infrastructure.orm_models import Project
from fastapi.testclient import TestClient
from inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER
from sqlalchemy import select

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
            session.add(
                Project(id=project_id, name="Callback test", slug=f"callback-{uuid.uuid4().hex}")
            )
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
        session.add(
            Project(id=project_id, name="Callback test", slug=f"callback-{uuid.uuid4().hex}")
        )
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
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 401


def test_callback_wrong_secret_returns_403(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers={INFERENCE_WEBHOOK_SECRET_HEADER: "wrong-secret"},
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 403


def test_callback_unconfigured_secret_returns_503(client: TestClient, monkeypatch):
    # Empty rather than deleted: settings also read `backend/core/.env`, falling
    # back to `.env.supabase`, so removing the process variable leaves whatever
    # the dotenv supplies and this asserts nothing in a configured checkout. An
    # env var overrides the file, and the route treats empty as unconfigured.
    monkeypatch.setenv("INFERENCE_WEBHOOK_SECRET", "")
    get_inference_settings.cache_clear()
    product_job_id, inference_job_id = _seed_waiting_job()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
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
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
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
    # The inference service's own message is the only diagnostic it sends; it has
    # to survive the hop, behind a stable prefix.
    assert job.error == f"{INFERENCE_FAILURE_ERROR}: weights not found in cache"
    assert job.completed_at is not None


def test_callback_failure_redacts_secret_shaped_detail(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    payload = _failed_payload(product_job_id=product_job_id, inference_job_id=inference_job_id)
    payload["error"] = "download failed: https://weights.example/model?token=s3cr3ttokenvalue000"

    response = client.post(CALLBACK_URL, headers=WEBHOOK_HEADERS, json=payload)
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.error is not None
    assert job.error.startswith(INFERENCE_FAILURE_ERROR)
    assert "s3cr3ttokenvalue000" not in job.error
    assert "weights.example" not in job.error


# --- Idempotency and validation ---
# Tests duplicate callbacks and mismatched job metadata. Does not enqueue new inference work.


def test_callback_on_terminal_job_is_idempotent(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    payload = _segment_done_payload(
        product_job_id=product_job_id, inference_job_id=inference_job_id
    )

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


def test_callback_after_cancel_skips_merge(client: TestClient):
    """Cancelled waiting jobs stay cancelled; document merge must not run."""
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.status = JobStatus.cancelled
        job.completed_at = datetime.now(UTC)
        job.updated_at = job.completed_at
        session.commit()
        part_id = job.document_part_id
        assert part_id is not None

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.cancelled
    assert job.result is None

    with sync_system_session() as session:
        lines = list(session.execute(select(Line).where(Line.part_id == part_id)).scalars().all())
    assert lines == []


def test_parallel_callback_replays_merge_once(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    payload = _segment_done_payload(
        product_job_id=product_job_id, inference_job_id=inference_job_id
    )

    def deliver() -> int:
        return client.post(CALLBACK_URL, headers=WEBHOOK_HEADERS, json=payload).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(lambda _unused: deliver(), range(2)))

    assert statuses == [204, 204]
    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.result is not None
    assert job.result["lines_count"] == 1


def test_callback_task_mismatch_returns_409(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job(job_type=JobType.transcribe)
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
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


def test_callback_rejects_unbound_inference_job(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.inference_job_id = None
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )

    assert response.status_code == 409
    assert _get_job(product_job_id).status == JobStatus.waiting


def test_callback_requires_waiting_job_state(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.status = JobStatus.running
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )

    assert response.status_code == 409
    assert _get_job(product_job_id).status == JobStatus.running


def test_callback_claimed_replay_is_not_merged_twice(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.callback_claimed_at = datetime.now(UTC)
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )

    assert response.status_code == 204
    job = _get_job(product_job_id)
    assert job.status == JobStatus.waiting
    assert job.result is None


# --- Merge/finalize atomicity ---
# Tests a raise after the document merge takes the merge down with it. Does not
# test the claim, which commits in its own transaction by design.


def test_finalize_failure_rolls_back_the_merged_lines(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Merge + finalize are one transaction: neither the lines nor a done job survive."""
    product_job_id, inference_job_id = _seed_waiting_job()
    part_id = _get_job(product_job_id).document_part_id
    assert part_id is not None

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("finalize exploded")

    monkeypatch.setattr(job_callback_service, "_mark_done_from_callback_sync", _boom)

    with pytest.raises(RuntimeError, match="finalize exploded"):
        client.post(
            CALLBACK_URL,
            headers=WEBHOOK_HEADERS,
            json=_segment_done_payload(
                product_job_id=product_job_id, inference_job_id=inference_job_id
            ),
        )

    with sync_system_session() as session:
        lines = list(session.execute(select(Line).where(Line.part_id == part_id)).scalars().all())
    assert lines == []

    job = _get_job(product_job_id)
    assert job.status != JobStatus.done
    assert job.result is None
    # The claim committed separately, so it still needs a compensating write -
    # otherwise the job hangs claimed until the stale-claim sweep notices.
    assert job.status == JobStatus.failed
    assert job.callback_claimed_at is None
    assert job.completed_at is not None


def test_transcribe_finalize_failure_rolls_back_the_transcription(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    product_job_id, inference_job_id, line_id = _seed_transcribe_waiting_job()

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("finalize exploded")

    monkeypatch.setattr(job_callback_service, "_mark_done_from_callback_sync", _boom)

    with pytest.raises(RuntimeError, match="finalize exploded"):
        client.post(
            CALLBACK_URL,
            headers=WEBHOOK_HEADERS,
            json=_transcribe_done_payload(
                product_job_id=product_job_id,
                inference_job_id=inference_job_id,
                line_id=line_id,
            ),
        )

    with sync_system_session() as session:
        transcriptions = list(
            session.execute(
                select(Transcription).where(Transcription.created_by_job_id == product_job_id)
            )
            .scalars()
            .all()
        )
    assert transcriptions == []
    assert _get_job(product_job_id).status == JobStatus.failed


def test_callback_unknown_job_returns_404(client: TestClient):
    product_job_id = uuid.uuid4()
    inference_job_id = uuid.uuid4()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 404
