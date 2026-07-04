"""Transcribe job integration tests — public HTTP behavior."""

from __future__ import annotations

import time
import uuid
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image


def _one_pixel_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


MINIMAL_PNG = _one_pixel_png()


def _documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def _create_document_part_with_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str]:
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Transcribe codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "kraken",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "kraken",
                },
            ]
        },
    )
    assert replace.status_code == 200
    return project_id, document_id, part_id


def _poll_job(
    client: TestClient,
    job_id: str,
    *,
    expect: str,
    headers: dict[str, str],
    timeout: float = 5.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        body = response.json()
        if body["status"] == expect:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status {expect!r} in {timeout}s")


def test_transcribe_job_creates_model_layer_and_leaves_ground_truth_empty(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    enqueue = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert enqueue.status_code == 202
    job_id = enqueue.json()["job_id"]

    job = _poll_job(client, job_id, expect="done", headers=owner_headers, timeout=8.0)
    assert job["type"] == "transcribe"
    assert [line["text"] for line in job["result"]["lines"]] == [
        "mock transcription 1",
        "mock transcription 2",
    ]
    assert [line["confidence"] for line in job["result"]["lines"]] == [0.91, 0.82]

    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers)
    assert layers.status_code == 200
    layer_body = layers.json()
    assert [layer["kind"] for layer in layer_body] == ["ground_truth", "model"]
    assert layer_body[1]["created_by_job_id"] == job_id

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    for line in lines.json():
        assert [entry["transcription_kind"] for entry in line["line_transcriptions"]] == ["model"]


def test_each_transcribe_job_creates_distinct_model_layer_without_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    job_ids = []
    for _ in range(2):
        enqueue = client.post(
            f"{base}/{document_id}/parts/{part_id}/transcribe",
            headers=owner_headers,
        )
        assert enqueue.status_code == 202
        job_ids.append(enqueue.json()["job_id"])
        _poll_job(client, job_ids[-1], expect="done", headers=owner_headers, timeout=8.0)

    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers)
    assert layers.status_code == 200
    layer_body = layers.json()
    assert [layer["kind"] for layer in layer_body] == ["ground_truth", "model", "model"]
    assert {layer["created_by_job_id"] for layer in layer_body[1:]} == set(job_ids)
    assert layer_body[1]["id"] != layer_body[2]["id"]

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    for line in lines.json():
        assert [entry["transcription_kind"] for entry in line["line_transcriptions"]] == [
            "model",
            "model",
        ]


def test_copy_model_layer_to_ground_truth_for_whole_document(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    enqueue = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert enqueue.status_code == 202
    _poll_job(
        client, enqueue.json()["job_id"], expect="done", headers=owner_headers, timeout=8.0
    )
    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers).json()
    model_layer_id = next(layer["id"] for layer in layers if layer["kind"] == "model")

    copy = client.post(
        f"{base}/{document_id}/transcriptions/{model_layer_id}/copy-to-ground-truth",
        headers=owner_headers,
        json={},
    )

    assert copy.status_code == 200
    assert copy.json()["copied_line_ids"]
    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    for line in lines.json():
        texts_by_kind = {
            entry["transcription_kind"]: entry["text"]
            for entry in line["line_transcriptions"]
        }
        assert texts_by_kind["ground_truth"] == texts_by_kind["model"]


def test_copy_model_layer_to_ground_truth_for_selected_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    enqueue = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert enqueue.status_code == 202
    _poll_job(
        client, enqueue.json()["job_id"], expect="done", headers=owner_headers, timeout=8.0
    )
    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers).json()
    model_layer_id = next(layer["id"] for layer in layers if layer["kind"] == "model")
    lines_before = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers
    ).json()
    selected_line_id = lines_before[0]["id"]

    copy = client.post(
        f"{base}/{document_id}/transcriptions/{model_layer_id}/copy-to-ground-truth",
        headers=owner_headers,
        json={"line_ids": [selected_line_id]},
    )

    assert copy.status_code == 200
    assert copy.json()["copied_line_ids"] == [selected_line_id]
    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    assert {
        entry["transcription_kind"] for entry in lines.json()[0]["line_transcriptions"]
    } == {"model", "ground_truth"}
    assert {
        entry["transcription_kind"] for entry in lines.json()[1]["line_transcriptions"]
    } == {"model"}


def test_patch_ground_truth_line_text_persists(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    ground_truth_layer_id = client.get(
        f"{base}/{document_id}/transcriptions", headers=owner_headers
    ).json()[0]["id"]
    line_id = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers
    ).json()[0]["id"]

    patch = client.patch(
        f"{base}/{document_id}/transcriptions/{ground_truth_layer_id}/lines/{line_id}",
        headers=owner_headers,
        json={"text": "curated ground truth"},
    )

    assert patch.status_code == 200
    assert patch.json()["transcription_kind"] == "ground_truth"
    assert patch.json()["text"] == "curated ground truth"
    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    first_line_entries = lines.json()[0]["line_transcriptions"]
    assert first_line_entries == [
        {
            "id": patch.json()["id"],
            "transcription_id": ground_truth_layer_id,
            "transcription_kind": "ground_truth",
            "text": "curated ground truth",
            "confidence": None,
        }
    ]


def test_patch_model_layer_line_text_is_rejected(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    enqueue = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert enqueue.status_code == 202
    _poll_job(
        client, enqueue.json()["job_id"], expect="done", headers=owner_headers, timeout=8.0
    )
    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers).json()
    model_layer_id = next(layer["id"] for layer in layers if layer["kind"] == "model")
    first_line = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers
    ).json()[0]

    patch = client.patch(
        f"{base}/{document_id}/transcriptions/{model_layer_id}/lines/{first_line['id']}",
        headers=owner_headers,
        json={"text": "attempted overwrite"},
    )

    assert patch.status_code == 409
    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    model_entry = next(
        entry
        for entry in lines.json()[0]["line_transcriptions"]
        if entry["transcription_kind"] == "model"
    )
    assert model_entry["text"] == "mock transcription 1"


def test_copy_from_unknown_layer_returns_not_found(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, _part_id = _create_document_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    copy = client.post(
        f"{base}/{document_id}/transcriptions/{uuid.uuid4()}/copy-to-ground-truth",
        headers=owner_headers,
        json={},
    )

    assert copy.status_code == 404
