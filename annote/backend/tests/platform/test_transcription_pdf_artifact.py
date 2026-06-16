"""Transcription PDF artifacts through the platform API."""

from __future__ import annotations

import io
import uuid
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader


def _documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def _png_bytes(width: int = 160, height: int = 90) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _pdf_reader(pdf_bytes: bytes) -> PdfReader:
    return PdfReader(io.BytesIO(pdf_bytes))


def _create_document_part_with_segments(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str, list[str]]:
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "PDF codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", _png_bytes(), "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    line_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": line_ids[0],
                    "order": 0,
                    "kind": "polygon",
                    "points": [[10, 10], [120, 10], [120, 30], [10, 30]],
                    "source": "manual",
                },
                {
                    "id": line_ids[1],
                    "order": 1,
                    "kind": "polygon",
                    "points": [[10, 45], [120, 45], [120, 65], [10, 65]],
                    "source": "manual",
                },
            ]
        },
    )
    assert replace.status_code == 200
    return project_id, document_id, part_id, line_ids


def test_member_generates_transcription_pdf_from_paired_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\nunused"},
    )
    assert import_response.status_code == 200
    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert pair.status_code == 200

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=owner_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    reader = _pdf_reader(response.content)
    assert len(reader.pages) == 1
    assert reader.pages[0].extract_text() == "alpha\n"
    assert reader.pages[0].mediabox.width == 160
    assert reader.pages[0].mediabox.height == 90


def test_member_generates_blank_same_size_transcription_pdf_without_paired_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=owner_headers,
    )

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
    reader = _pdf_reader(response.content)
    assert len(reader.pages) == 1
    assert (reader.pages[0].extract_text() or "").strip() == ""
    assert reader.pages[0].mediabox.width == 160
    assert reader.pages[0].mediabox.height == 90


def test_outsider_cannot_generate_transcription_pdf(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcription-pdf",
        headers=outsider_headers,
    )

    assert response.status_code in (403, 404)
