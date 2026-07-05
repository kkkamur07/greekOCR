"""Segment pipeline — enqueue segment jobs and merge canonical layout."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

ANNOTE_ROOT = Path(__file__).resolve().parents[3]
SEGMENT_IMAGE_PATH = (
    ANNOTE_ROOT
    / "data/manuscripts/pages/Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_6.jpeg"
)
SEGMENT_IMAGE_BYTES = SEGMENT_IMAGE_PATH.read_bytes()


def _documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def _create_document_with_part(client, owner_headers, owner_project) -> tuple[str, str, str]:
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Segmented codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": (SEGMENT_IMAGE_PATH.name, SEGMENT_IMAGE_BYTES, "image/jpeg")},
    )
    assert upload.status_code == 201
    return project_id, document_id, upload.json()["id"]


def _poll_job(
    client, job_id: str, *, expect: str, headers: dict[str, str], timeout: float = 5.0
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


@pytest.mark.integration
def test_member_can_enqueue_segment_job_and_poll_result(client, owner_headers, owner_project):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={"use_otsu_refinement": True, "otsu_sphere_radius": 6},
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    uuid.UUID(job_id)

    job = _poll_job(client, job_id, expect="done", headers=owner_headers, timeout=30.0)
    assert job["type"] == "segment"
    assert job["document_id"] == document_id
    assert job["document_part_id"] == part_id
    assert job["payload"]["ml_params"]["use_otsu_refinement"] is True
    assert job["payload"]["ml_params"]["otsu_sphere_radius"] == 6.0
    assert job["result"]["blocks_count"] == 1
    assert job["result"]["lines_count"] >= 1


@pytest.mark.integration
def test_segment_then_transcribe_jobs_run_real_ml_queue(client, owner_headers, owner_project):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    segment = client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={},
    )
    assert segment.status_code == 202
    segment_job = _poll_job(
        client,
        segment.json()["job_id"],
        expect="done",
        headers=owner_headers,
        timeout=30.0,
    )
    assert segment_job["result"]["lines_count"] >= 1

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    line_count = len(lines.json())
    assert line_count == segment_job["result"]["lines_count"]

    transcribe = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert transcribe.status_code == 202
    transcribe_job = _poll_job(
        client,
        transcribe.json()["job_id"],
        expect="done",
        headers=owner_headers,
        timeout=120.0,
    )
    assert transcribe_job["type"] == "transcribe"
    assert len(transcribe_job["result"]["lines"]) == line_count

    transcribed_lines = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert transcribed_lines.status_code == 200
    for line in transcribed_lines.json():
        assert any(
            entry["transcription_kind"] == "model"
            for entry in line["line_transcriptions"]
        )


@pytest.mark.integration
def test_segment_merge_preserves_manual_lines_and_prunes_machine_lines(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    seed = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[100, 100], [120, 100], [120, 110], [100, 110]],
                    "source": "manual",
                    "approved_text": "manual text survives",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 50], [20, 50], [20, 58], [0, 58]],
                    "source": "kraken",
                    "approved_text": "obsolete machine text",
                },
            ]
        },
    )
    assert seed.status_code == 200
    manual_line_id = seed.json()[0]["id"]
    machine_line_id = seed.json()[1]["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/segment",
        headers=owner_headers,
        json={},
    )
    assert response.status_code == 202
    job = _poll_job(client, response.json()["job_id"], expect="done", headers=owner_headers)
    assert job["result"]["preserved_manual_lines"] == 1
    assert job["result"]["pruned_lines"] == 1

    listed = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert listed.status_code == 200
    lines = listed.json()
    assert any(line["id"] == manual_line_id for line in lines)
    assert not any(line["id"] == machine_line_id for line in lines)
    manual = next(line for line in lines if line["id"] == manual_line_id)
    assert manual["points"] == [[100, 100], [120, 100], [120, 110], [100, 110]]
    assert manual["manual_geometry"] is True
    assert manual["line_transcriptions"][0]["text"] == "manual text survives"
    assert sum(1 for line in lines if line["source"] == "kraken") >= 1


@pytest.mark.integration
def test_layout_edit_sets_manual_geometry_and_reset_clears_selected_lines(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    block = client.post(
        f"{base}/{document_id}/parts/{part_id}/blocks",
        headers=owner_headers,
        json={"order": 0, "box": {"points": [[0, 0], [50, 0], [50, 20], [0, 20]]}},
    )
    assert block.status_code == 201
    block_id = block.json()["id"]

    line = client.post(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "order": 0,
            "kind": "polygon",
            "points": [[1, 1], [20, 1], [20, 8], [1, 8]],
            "block_id": block_id,
        },
    )
    assert line.status_code == 201
    line_id = line.json()["id"]
    assert line.json()["manual_geometry"] is True

    patched_block = client.patch(
        f"{base}/{document_id}/parts/{part_id}/blocks/{block_id}",
        headers=owner_headers,
        json={"box": {"points": [[0, 0], [60, 0], [60, 22], [0, 22]]}},
    )
    assert patched_block.status_code == 200
    assert patched_block.json()["manual_geometry"] is True
    assert patched_block.json()["box"]["points"][1] == [60, 0]

    patched_line = client.patch(
        f"{base}/{document_id}/parts/{part_id}/lines/{line_id}",
        headers=owner_headers,
        json={"baseline": {"points": [[2, 2], [22, 2]]}},
    )
    assert patched_line.status_code == 200
    assert patched_line.json()["baseline"] == {"points": [[2, 2], [22, 2]]}
    assert patched_line.json()["manual_geometry"] is True

    reset = client.post(
        f"{base}/{document_id}/parts/{part_id}/layout/reset",
        headers=owner_headers,
        json={"line_ids": [line_id]},
    )
    assert reset.status_code == 200
    reset_line = next(line for line in reset.json()["lines"] if line["id"] == line_id)
    assert reset_line["manual_geometry"] is False
