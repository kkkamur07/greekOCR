"""Public access policy — anonymous read of published documents only."""

import uuid

import pytest

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
    b"\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def published_document(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Public codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    publish = client.patch(
        f"{base}/{document_id}",
        headers=owner_headers,
        json={"workflow": "published"},
    )
    assert publish.status_code == 200
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]
    return {
        "project_id": project_id,
        "document_id": document_id,
        "part_id": part_id,
    }


@pytest.mark.integration
def test_anonymous_can_read_published_document(client, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    url = f"/public/projects/{project_id}/documents/{document_id}"

    response = client.get(url)
    assert response.status_code == 200
    body = response.json()
    assert body["workflow"] == "published"
    assert len(body["parts"]) == 1


@pytest.mark.integration
def test_anonymous_cannot_read_draft_document(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Secret draft"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    response = client.get(f"/public/projects/{project_id}/documents/{document_id}")
    assert response.status_code == 404


@pytest.mark.integration
def test_anonymous_cannot_mutate_published_document(client, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    member_url = f"/projects/{project_id}/documents/{document_id}"

    patch = client.patch(member_url, json={"name": "Hacked"})
    assert patch.status_code == 401

    upload = client.post(
        f"{member_url}/parts",
        files={"file": ("x.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 401


@pytest.mark.integration
def test_member_can_still_edit_published_document(client, owner_headers, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    url = f"/projects/{project_id}/documents/{document_id}"

    patch = client.patch(
        url, headers=owner_headers, json={"name": "Published but editable"}
    )
    assert patch.status_code == 200
    assert patch.json()["name"] == "Published but editable"
    assert patch.json()["workflow"] == "published"


@pytest.mark.integration
def test_outsider_can_read_published_via_public_route(
    client, outsider_headers, published_document
):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    member_url = f"/projects/{project_id}/documents/{document_id}"
    public_url = f"/public/projects/{project_id}/documents/{document_id}"

    denied = client.get(member_url, headers=outsider_headers)
    assert denied.status_code == 403

    allowed = client.get(public_url)
    assert allowed.status_code == 200


@pytest.mark.integration
def test_anonymous_can_read_published_part_media(client, published_document):
    part_id = published_document["part_id"]
    response = client.get(f"/public/media/parts/{part_id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


@pytest.mark.integration
def test_anonymous_gets_layout_and_transcriptions(client, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    base = f"/public/projects/{project_id}/documents/{document_id}"

    layout = client.get(f"{base}/layout")
    assert layout.status_code == 200
    assert layout.json() == {"blocks": [], "lines": []}

    layers = client.get(f"{base}/transcriptions")
    assert layers.status_code == 200
    assert layers.json()[0]["kind"] == "ground_truth"
