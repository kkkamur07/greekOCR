"""Document and part integration tests — real Postgres (kalamos)."""

import uuid

import pytest

# Minimal valid 1×1 PNG
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
    b"\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe"
    b"\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def _create_document_with_part(client, owner_headers, owner_project) -> tuple[str, str, str]:
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Codex with page"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    return project_id, document_id, upload.json()["id"]


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
    assert doc["part_count"] == 0

    listed = client.get(base, headers=owner_headers)
    assert listed.status_code == 200
    listed_doc = next(d for d in listed.json() if d["id"] == document_id)
    assert listed_doc["part_count"] == 0

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

    listed = client.get(base, headers=owner_headers)
    listed_doc = next(d for d in listed.json() if d["id"] == document_id)
    assert listed_doc["part_count"] == 2

    project = client.get(f"/projects/{project_id}", headers=owner_headers)
    assert project.json()["document_count"] >= 1

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
    assert dashboard.json()["part_count"] == 2
    assert len(dashboard.json()["parts"]) == 2

    delete_part = client.delete(
        f"{base}/{document_id}/parts/{part_a['id']}",
        headers=owner_headers,
    )
    assert delete_part.status_code == 204

    after = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert after.json()["part_count"] == 1
    assert len(after.json()["parts"]) == 1


@pytest.mark.integration
def test_upload_part_rejects_non_image_bytes(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = _documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Invalid upload"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    response = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("not-image.txt", b"<script>alert(1)</script>", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "VALIDATION_ERROR",
        "message": "Uploaded file is not a valid image",
    }


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


@pytest.mark.integration
def test_document_gets_canonical_ground_truth_transcription_layer(
    client, owner_headers, owner_project
) -> None:
    project_id = owner_project["id"]
    base = _documents_url(project_id)

    create = client.post(base, headers=owner_headers, json={"name": "Layered codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    layers = client.get(f"{base}/{document_id}/transcriptions", headers=owner_headers)
    assert layers.status_code == 200
    body = layers.json()
    assert len(body) == 1
    assert body[0]["name"] == "Ground truth"
    assert body[0]["kind"] == "ground_truth"


@pytest.mark.integration
def test_part_review_status_defaults_unreviewed_and_can_toggle(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    read = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert read.status_code == 200
    assert read.json()["parts"][0]["reviewed"] is False

    reviewed = client.patch(
        f"{base}/{document_id}/parts/{part_id}",
        headers=owner_headers,
        json={"reviewed": True},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["reviewed"] is True

    unreviewed = client.patch(
        f"{base}/{document_id}/parts/{part_id}",
        headers=owner_headers,
        json={"reviewed": False},
    )
    assert unreviewed.status_code == 200
    assert unreviewed.json()["reviewed"] is False


@pytest.mark.integration
def test_replace_part_lines_persists_segment_geometry_and_approved_text(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

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
                    "approved_text": "approved ground truth",
                },
                {
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
    lines = replace.json()
    assert [line["order"] for line in lines] == [0, 1]
    uuid.UUID(lines[0]["id"])
    assert lines[0]["kind"] == "polygon"
    assert lines[0]["source"] == "manual"
    assert lines[0]["line_transcriptions"][0]["transcription_kind"] == "ground_truth"
    assert lines[0]["line_transcriptions"][0]["text"] == "approved ground truth"
    assert lines[1]["source"] == "kraken"
    assert lines[1]["kraken_ceiling"] == [[-1, 9], [11, 9], [11, 16], [-1, 16]]
    assert lines[1]["line_transcriptions"] == []
    assert lines[0]["baseline"]["points"] == [[0, 5], [10, 5]]
    assert lines[1]["baseline"]["points"] == [[0, 15], [10, 15]]

    listed = client.get(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
    )
    assert listed.status_code == 200
    assert listed.json() == lines


@pytest.mark.integration
def test_replace_part_lines_preserves_existing_kraken_baseline(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)
    kraken_baseline = {"points": [[1, 7], [5, 7.5], [9, 7]]}
    seed = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 8], [0, 8]],
                    "source": "kraken",
                    "baseline": kraken_baseline,
                    "mask": {"points": [[0, 0], [10, 0], [10, 8], [0, 8]]},
                }
            ]
        },
    )
    assert seed.status_code == 200
    line_id = seed.json()[0]["id"]

    updated = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": line_id,
                    "order": 0,
                    "kind": "polygon",
                    "points": [[1, 1], [11, 1], [11, 9], [1, 9]],
                    "source": "kraken",
                }
            ]
        },
    )
    assert updated.status_code == 200
    assert updated.json()[0]["baseline"] == kraken_baseline


@pytest.mark.integration
def test_replace_part_lines_rejects_client_selected_id_for_new_line(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = _documents_url(project_id)

    response = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": str(uuid.uuid4()),
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "source": "manual",
                },
            ]
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "New line ids are server-generated"


@pytest.mark.integration
def test_document_line_transcription_contract_is_in_openapi(client):
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    components = schema.json()["components"]["schemas"]

    assert "LineResponse" in components
    assert "LineTranscriptionResponse" in components
    assert "TranscriptionLayerResponse" in components
    part_properties = components["DocumentPartResponse"]["properties"]
    assert part_properties["reviewed"]["type"] == "boolean"
