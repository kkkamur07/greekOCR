"""Page transcription Pairing workflow API tests."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from tests.platform.test_documents import MINIMAL_PNG


def _documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def _create_document_part_with_segments(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str, list[str]]:
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Pairing codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", MINIMAL_PNG, "image/png")},
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
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "manual",
                },
                {
                    "id": line_ids[1],
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "manual",
                },
            ]
        },
    )
    assert replace.status_code == 200
    return project_id, document_id, part_id, line_ids


def test_import_page_transcription_splits_candidate_lines_without_ground_truth(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\n\nbeta\n"},
    )

    assert import_response.status_code == 200
    assert import_response.json()["text_lines"] == [
        {"order": 0, "text": "alpha", "paired_line_id": None},
        {"order": 1, "text": "beta", "paired_line_id": None},
    ]
    assert import_response.json()["pairing_progress"] == {
        "paired_lines": 0,
        "total_lines": 2,
        "percent": 0,
    }

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    assert [line["line_transcriptions"] for line in lines.json()] == [[], []]

    reloaded = client.get(
        f"{base}/{document_id}/parts/{part_id}/pairing",
        headers=owner_headers,
    )
    assert reloaded.status_code == 200
    assert reloaded.json() == import_response.json()


def test_pair_candidate_text_line_creates_ground_truth_and_updates_progress(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\nbeta"},
    )
    assert import_response.status_code == 200

    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 1},
    )

    assert pair.status_code == 200
    assert pair.json()["text_lines"] == [
        {"order": 0, "text": "alpha", "paired_line_id": None},
        {"order": 1, "text": "beta", "paired_line_id": line_ids[0]},
    ]
    assert pair.json()["pairing_progress"] == {
        "paired_lines": 1,
        "total_lines": 2,
        "percent": 50,
    }

    lines = client.get(f"{base}/{document_id}/parts/{part_id}/lines", headers=owner_headers)
    assert lines.status_code == 200
    assert lines.json()[0]["line_transcriptions"][0]["text"] == "beta"
    assert lines.json()[0]["line_transcriptions"][0]["transcription_kind"] == "ground_truth"
    assert lines.json()[1]["line_transcriptions"] == []


def test_direct_ground_truth_text_edit_counts_toward_pairing_progress(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    ground_truth_id = client.get(
        f"{base}/{document_id}/transcriptions", headers=owner_headers
    ).json()[0]["id"]

    edit = client.patch(
        f"{base}/{document_id}/transcriptions/{ground_truth_id}/lines/{line_ids[0]}",
        headers=owner_headers,
        json={"text": "typed approved text"},
    )
    assert edit.status_code == 200

    pairing = client.get(
        f"{base}/{document_id}/parts/{part_id}/pairing",
        headers=owner_headers,
    )
    assert pairing.status_code == 200
    assert pairing.json()["pairing_progress"] == {
        "paired_lines": 1,
        "total_lines": 2,
        "percent": 50,
    }


def test_part_review_status_stays_independent_from_partial_pairing_and_text_edits(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    import_response = client.put(
        f"{base}/{document_id}/parts/{part_id}/page-transcription",
        headers=owner_headers,
        json={"text": "alpha\nbeta"},
    )
    assert import_response.status_code == 200
    pair = client.post(
        f"{base}/{document_id}/parts/{part_id}/pairings",
        headers=owner_headers,
        json={"line_id": line_ids[0], "text_line_order": 0},
    )
    assert pair.status_code == 200
    assert pair.json()["pairing_progress"] == {
        "paired_lines": 1,
        "total_lines": 2,
        "percent": 50,
    }

    reviewed = client.patch(
        f"{base}/{document_id}/parts/{part_id}",
        headers=owner_headers,
        json={"reviewed": True},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["reviewed"] is True

    reloaded = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert reloaded.status_code == 200
    assert reloaded.json()["parts"][0]["reviewed"] is True

    ground_truth_id = client.get(
        f"{base}/{document_id}/transcriptions", headers=owner_headers
    ).json()[0]["id"]
    edit = client.patch(
        f"{base}/{document_id}/transcriptions/{ground_truth_id}/lines/{line_ids[0]}",
        headers=owner_headers,
        json={"text": "corrected alpha"},
    )
    assert edit.status_code == 200

    after_edit = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert after_edit.status_code == 200
    assert after_edit.json()["parts"][0]["reviewed"] is True


def test_outsider_cannot_change_part_review_status(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_document_part_with_segments(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    denied = client.patch(
        f"{base}/{document_id}/parts/{part_id}",
        headers=outsider_headers,
        json={"reviewed": True},
    )

    assert denied.status_code in (403, 404)
    reloaded = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert reloaded.status_code == 200
    assert reloaded.json()["parts"][0]["reviewed"] is False
