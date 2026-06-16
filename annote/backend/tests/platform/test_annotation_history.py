"""Annotation history API tests."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from tests.platform.test_documents import MINIMAL_PNG


def _documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def _create_part_with_lines(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> tuple[str, str, str, list[str]]:
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Recoverable codex"})
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
                    "approved_text": "alpha",
                },
                {
                    "id": line_ids[1],
                    "order": 1,
                    "kind": "rectangle",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "kraken",
                    "source_metadata": {"model": "kraken:blla"},
                    "kraken_ceiling": [[-1, 9], [11, 9], [11, 16], [-1, 16]],
                },
            ]
        },
    )
    assert replace.status_code == 200
    return project_id, document_id, part_id, line_ids


def test_member_can_create_and_list_compact_history_snapshots(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"

    created = client.post(history_url, headers=owner_headers)

    assert created.status_code == 201
    snapshot = created.json()
    assert snapshot["part_id"] == part_id
    assert snapshot["line_count"] == 2
    assert snapshot["paired_line_count"] == 1
    assert snapshot["state"]["lines"] == [
        {
            "id": line_ids[0],
            "order": 0,
            "kind": "polygon",
            "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "source": "manual",
            "source_metadata": None,
            "kraken_ceiling": None,
            "approved_text": "alpha",
        },
        {
            "id": line_ids[1],
            "order": 1,
            "kind": "rectangle",
            "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
            "source": "kraken",
            "source_metadata": {"model": "kraken:blla"},
            "kraken_ceiling": [[-1, 9], [11, 9], [11, 16], [-1, 16]],
            "approved_text": None,
        },
    ]
    assert "image_key" not in snapshot["state"]
    assert "image_url" not in snapshot["state"]
    assert "exports" not in snapshot["state"]
    assert "events" not in snapshot["state"]

    listed = client.get(history_url, headers=owner_headers)

    assert listed.status_code == 200
    assert listed.json() == [snapshot]


def test_member_can_restore_snapshot_to_replace_current_page_annotations(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"
    snapshot_id = client.post(history_url, headers=owner_headers).json()["id"]

    replace_current = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": line_ids[0],
                    "order": 0,
                    "kind": "polygon",
                    "points": [[100, 100], [110, 100], [110, 105], [100, 105]],
                    "source": "manual",
                    "approved_text": "changed",
                },
                {
                    "id": str(uuid.uuid4()),
                    "order": 1,
                    "kind": "polygon",
                    "points": [[200, 200], [210, 200], [210, 205], [200, 205]],
                    "source": "manual",
                    "approved_text": "new line",
                },
            ]
        },
    )
    assert replace_current.status_code == 200

    restored = client.post(
        f"{history_url}/{snapshot_id}/restore",
        headers=owner_headers,
    )

    assert restored.status_code == 200
    reloaded = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert reloaded.status_code == 200
    restored_lines = reloaded.json()
    assert [line["id"] for line in restored_lines] == line_ids
    assert [line["points"] for line in restored_lines] == [
        [[0, 0], [10, 0], [10, 5], [0, 5]],
        [[0, 10], [10, 10], [10, 15], [0, 15]],
    ]
    assert restored_lines[0]["line_transcriptions"][0]["text"] == "alpha"
    assert restored_lines[1]["line_transcriptions"] == []
    assert restored.json() == restored_lines


def test_non_member_cannot_create_list_or_restore_history_snapshots(
    client: TestClient,
    owner_headers: dict[str, str],
    outsider_headers: dict[str, str],
    owner_project: dict,
) -> None:
    project_id, document_id, part_id, _line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"
    snapshot_id = client.post(history_url, headers=owner_headers).json()["id"]

    create = client.post(history_url, headers=outsider_headers)
    listed = client.get(history_url, headers=outsider_headers)
    restored = client.post(
        f"{history_url}/{snapshot_id}/restore",
        headers=outsider_headers,
    )

    assert create.status_code in (403, 404)
    assert listed.status_code in (403, 404)
    assert restored.status_code in (403, 404)


def test_history_snapshot_retention_keeps_only_latest_twenty(
    client: TestClient, owner_headers: dict[str, str], owner_project: dict
) -> None:
    project_id, document_id, part_id, _line_ids = _create_part_with_lines(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    history_url = f"{base}/{document_id}/parts/{part_id}/history"

    snapshot_ids = []
    for index in range(21):
        created = client.post(history_url, headers=owner_headers)
        assert created.status_code == 201
        snapshot_ids.append(created.json()["id"])

    listed = client.get(history_url, headers=owner_headers)

    assert listed.status_code == 200
    listed_ids = [snapshot["id"] for snapshot in listed.json()]
    assert len(listed_ids) == 20
    assert snapshot_ids[0] not in listed_ids
    assert snapshot_ids[-1] in listed_ids

    pruned_restore = client.post(
        f"{history_url}/{snapshot_ids[0]}/restore",
        headers=owner_headers,
    )
    assert pruned_restore.status_code == 404
