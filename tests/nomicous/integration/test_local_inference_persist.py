"""Local inference persist endpoints — browser-orchestrated results into Postgres."""

from __future__ import annotations

import pytest

from tests.nomicous.integration.helpers import documents_url
from tests.nomicous.integration.test_documents import _create_document_with_part


@pytest.mark.integration
def test_persist_local_transcribe_writes_model_transcription(client, owner_headers, owner_project):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "manual",
                }
            ]
        },
    )
    assert replace.status_code == 200
    line_id = replace.json()[0]["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/local-inference/transcribe",
        headers=owner_headers,
        json={
            "registry_model_id": "greek-calamari-v1",
            "registry_tag": "stable",
            "lines": [
                {
                    "line_id": line_id,
                    "text": "Αβ",
                    "confidence": 0.91,
                    "character_confidences": [
                        {"char": "Α", "confidence": 0.93},
                        {"char": "β", "confidence": 0.89},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["job_id"]
    assert body["transcription_id"]
    assert body["lines"][0]["line_id"] == line_id
    assert body["lines"][0]["text"] == "Αβ"
    assert body["lines"][0]["confidence"] == 0.91

    listed = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert listed.status_code == 200
    line = listed.json()[0]
    model_layers = [
        layer
        for layer in line["line_transcriptions"]
        if layer["transcription_kind"] == "model"
    ]
    assert len(model_layers) == 1
    assert model_layers[0]["text"] == "Αβ"
    assert model_layers[0]["confidence"] == 0.91

    jobs = client.get(f"/projects/{project_id}/jobs", headers=owner_headers)
    assert jobs.status_code == 200
    listed_jobs = jobs.json()["items"]
    local_job = next(job for job in listed_jobs if job["id"] == body["job_id"])
    assert local_job["type"] == "transcribe"
    assert local_job["status"] == "done"
    assert local_job["execution"] == "local"


@pytest.mark.integration
def test_persist_local_segment_preserves_manual_geometry(client, owner_headers, owner_project):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "manual",
                    "approved_text": "manual line",
                },
                {
                    "order": 1,
                    "kind": "rectangle",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "kraken",
                    "source_metadata": {"model": "kraken:blla"},
                },
            ]
        },
    )
    assert replace.status_code == 200
    manual_line_id = replace.json()[0]["id"]

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/local-inference/segment",
        headers=owner_headers,
        json={
            "registry_model_id": "greek-kraken-segment-v1",
            "registry_tag": "stable",
            "output": {
                "blocks": [],
                "lines": [
                    {
                        "external_id": "l-new",
                        "order": 0,
                        "baseline": {"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                        "points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                    }
                ],
            },
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["job_id"]
    assert body["lines_count"] == 1
    assert body["added_lines"] == 1
    assert body["pruned_lines"] == 1
    assert body["preserved_manual_lines"] == 1

    listed = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert listed.status_code == 200
    lines = listed.json()
    assert len(lines) == 2
    manual = next(line for line in lines if line["id"] == manual_line_id)
    assert manual["source"] == "manual"
    assert manual["line_transcriptions"][0]["text"] == "manual line"
    machine = next(line for line in lines if line["id"] != manual_line_id)
    assert machine["source"] == "kraken"

    jobs = client.get(f"/projects/{project_id}/jobs", headers=owner_headers)
    assert jobs.status_code == 200
    local_job = next(job for job in jobs.json()["items"] if job["id"] == body["job_id"])
    assert local_job["type"] == "segment"
    assert local_job["execution"] == "local"
