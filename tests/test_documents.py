"""Document and part integration tests — real Postgres (kalamos)."""

import uuid

import pytest

# Minimal valid 1×1 PNG
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
    b"\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


@pytest.mark.integration
def test_member_create_list_read_update_delete_document(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = _documents_url(project_id)

    create = client.post(base, headers=owner_headers, json={"name": "Codex A"})
    assert create.status_code == 201
    doc = create.json()
    document_id = doc["id"]
    assert doc["name"] == "Codex A"
    assert doc["workflow"] == "draft"

    listed = client.get(base, headers=owner_headers)
    assert listed.status_code == 200
    assert any(d["id"] == document_id for d in listed.json())

    read = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert read.status_code == 200
    assert read.json()["parts"] == []

    update = client.patch(
        f"{base}/{document_id}",
        headers=owner_headers,
        json={"name": "Renamed codex", "workflow": "published"},
    )
    assert update.status_code == 200
    assert update.json()["workflow"] == "published"

    delete = client.delete(f"{base}/{document_id}", headers=owner_headers)
    assert delete.status_code == 204

    gone = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert gone.status_code == 404


@pytest.mark.integration
def test_archived_document_hidden_from_default_list(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = _documents_url(project_id)

    create = client.post(base, headers=owner_headers, json={"name": "To archive"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    archive = client.patch(
        f"{base}/{document_id}",
        headers=owner_headers,
        json={"workflow": "archived"},
    )
    assert archive.status_code == 200
    assert archive.json()["workflow"] == "archived"

    default_list = client.get(base, headers=owner_headers)
    assert default_list.status_code == 200
    assert not any(d["id"] == document_id for d in default_list.json())

    with_archived = client.get(f"{base}?include_archived=true", headers=owner_headers)
    assert with_archived.status_code == 200
    assert any(d["id"] == document_id for d in with_archived.json())


@pytest.mark.integration
def test_outsider_cannot_access_documents(client, owner_headers, outsider_headers, owner_project):
    project_id = owner_project["id"]
    base = _documents_url(project_id)

    create = client.post(base, headers=owner_headers, json={"name": "Private codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    read = client.get(f"{base}/{document_id}", headers=outsider_headers)
    assert read.status_code in (403, 404)

    mutate = client.patch(
        f"{base}/{document_id}",
        headers=outsider_headers,
        json={"name": "Hijacked"},
    )
    assert mutate.status_code in (403, 404)


@pytest.mark.integration
def test_upload_reorder_delete_part_and_serve_media(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = _documents_url(project_id)

    create = client.post(base, headers=owner_headers, json={"name": "With parts"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload_a = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio-a.png", MINIMAL_PNG, "image/png")},
    )
    assert upload_a.status_code == 201
    part_a = upload_a.json()
    assert part_a["order"] == 0
    assert part_a["image_url"] == f"/media/parts/{part_a['id']}"

    upload_b = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio-b.png", MINIMAL_PNG, "image/png")},
    )
    assert upload_b.status_code == 201
    part_b = upload_b.json()
    assert part_b["order"] == 1

    reorder = client.patch(
        f"{base}/{document_id}/parts/reorder",
        headers=owner_headers,
        json={"part_ids": [part_b["id"], part_a["id"]]},
    )
    assert reorder.status_code == 200
    orders = [p["order"] for p in reorder.json()]
    assert orders == [0, 1]
    ids = [p["id"] for p in reorder.json()]
    assert ids[0] == part_b["id"]

    media = client.get(part_a["image_url"], headers=owner_headers)
    assert media.status_code == 200
    assert media.content == MINIMAL_PNG

    dashboard = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert dashboard.status_code == 200
    assert len(dashboard.json()["parts"]) == 2

    delete_part = client.delete(
        f"{base}/{document_id}/parts/{part_a['id']}",
        headers=owner_headers,
    )
    assert delete_part.status_code == 204

    after = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert len(after.json()["parts"]) == 1


@pytest.mark.integration
def test_reorder_rejects_duplicate_part_ids(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Reorder dup"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]

    reorder = client.patch(
        f"{base}/{document_id}/parts/reorder",
        headers=owner_headers,
        json={"part_ids": [part_id, part_id]},
    )
    assert reorder.status_code == 422


@pytest.mark.integration
def test_patch_null_name_returns_422(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Named"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    response = client.patch(
        f"{base}/{document_id}",
        headers=owner_headers,
        json={"name": None},
    )
    assert response.status_code == 422
