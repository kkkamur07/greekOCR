"""ML end-to-end — real async segment/transcribe via inference callbacks."""

from __future__ import annotations

import uuid
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.fixtures.paths import MINIMAL_PNG, segment_page_bytes
from tests.nomicous.integration.helpers import documents_url, poll_job


def _compact_segment_page_bytes() -> bytes:
    """Synthetic page with a few ink lines — fast segment + transcribe."""
    image = Image.new("RGB", (128, 128), color="white")
    pixels = image.load()
    for y in range(24, 110, 28):
        for x in range(12, 116):
            pixels[x, y] = (0, 0, 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _create_document_with_part_image(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
    *,
    filename: str,
    image_bytes: bytes,
    content_type: str,
    document_name: str,
) -> tuple[str, str, str]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": document_name})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": (filename, image_bytes, content_type)},
    )
    assert upload.status_code == 201
    return project_id, document_id, upload.json()["id"]


def _create_document_with_segment_page(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str]:
    page_bytes = segment_page_bytes()
    return _create_document_with_part_image(
        client,
        owner_headers,
        owner_project,
        filename="segment_page.jpeg",
        image_bytes=page_bytes,
        content_type="image/jpeg",
        document_name="ML E2E codex",
    )


def _create_document_with_compact_page(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str]:
    return _create_document_with_part_image(
        client,
        owner_headers,
        owner_project,
        filename="compact_page.png",
        image_bytes=_compact_segment_page_bytes(),
        content_type="image/png",
        document_name="ML chain codex",
    )


def _create_document_with_empty_part(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str]:
    return _create_document_with_part_image(
        client,
        owner_headers,
        owner_project,
        filename="page.png",
        image_bytes=MINIMAL_PNG,
        content_type="image/png",
        document_name="Empty layout codex",
    )


# --- Segment E2E (ML lane) ---
# Tests async segment job through real inference callback. Does not stub run_job.


@pytest.mark.ml
@pytest.mark.integration
def test_segment_job_completes_via_callback(
    platform_http_client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_with_segment_page(
        platform_http_client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    response = platform_http_client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    uuid.UUID(job_id)

    job = poll_job(
        platform_http_client, job_id, expect_status="done", headers=owner_headers, timeout=60.0
    )
    assert job["type"] == "segment"
    assert job["result"]["lines_count"] >= 1

    lines = platform_http_client.get(
        f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers
    )
    assert lines.status_code == 200
    assert len(lines.json()) >= 1


# --- Segment then transcribe chain (ML lane) ---
# Tests full layout + OCR pipeline end to end. Does not test failure recovery.


@pytest.mark.ml
@pytest.mark.integration
def test_segment_then_transcribe_chain(
    platform_http_client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_with_compact_page(
        platform_http_client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    segment = platform_http_client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={},
    )
    assert segment.status_code == 202
    segment_job = poll_job(
        platform_http_client,
        segment.json()["job_id"],
        expect_status="done",
        headers=owner_headers,
        timeout=60.0,
    )
    line_count = segment_job["result"]["lines_count"]
    assert line_count >= 1

    transcribe = platform_http_client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert transcribe.status_code == 202
    transcribe_job = poll_job(
        platform_http_client,
        transcribe.json()["job_id"],
        expect_status="done",
        headers=owner_headers,
        timeout=60.0,
    )
    assert transcribe_job["type"] == "transcribe"
    assert len(transcribe_job["result"]["lines"]) == line_count

    lines = platform_http_client.get(
        f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers
    )
    assert lines.status_code == 200
    for line in lines.json():
        assert any(entry["transcription_kind"] == "model" for entry in line["line_transcriptions"])


# --- Transcribe preconditions ---
# Tests transcribe is rejected without layout lines. Does not run inference.


@pytest.mark.ml
@pytest.mark.integration
def test_transcribe_without_lines_fails(
    platform_http_client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_with_empty_part(
        platform_http_client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    response = platform_http_client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )

    assert response.status_code == 409
    assert "without layout lines" in response.json()["error"]["message"]
