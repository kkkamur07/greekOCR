"""Public access policy — anonymous read of published documents only."""

import pytest

from tests.fixtures.paths import MINIMAL_PNG


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


# --- Anonymous read of published documents ---
# Tests public document detail for published workflow. Does not allow draft access.


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
    assert body["parts"][0]["image_url"] == f"/public/media/parts/{published_document['part_id']}"


# --- Draft documents stay private ---
# Tests anonymous users get 404 for draft workflow. Does not test member-route access.


@pytest.mark.integration
def test_anonymous_cannot_read_draft_document(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = f"/projects/{project_id}/documents"
    create = client.post(base, headers=owner_headers, json={"name": "Secret draft"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    response = client.get(f"/public/projects/{project_id}/documents/{document_id}")
    assert response.status_code == 404


# --- Anonymous mutations blocked ---
# Tests unauthenticated users cannot change published documents. Does not test member routes.


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


# --- Members can still edit published ---
# Tests owners retain mutate access after publish. Does not test collaborator permissions.


@pytest.mark.integration
def test_member_can_still_edit_published_document(client, owner_headers, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    url = f"/projects/{project_id}/documents/{document_id}"

    patch = client.patch(url, headers=owner_headers, json={"name": "Published but editable"})
    assert patch.status_code == 200
    assert patch.json()["name"] == "Published but editable"
    assert patch.json()["workflow"] == "published"


# --- Outsider vs public route ---
# Tests outsiders use /public for read access. Does not grant member-route access.


@pytest.mark.integration
def test_outsider_can_read_published_via_public_route(client, outsider_headers, published_document):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    member_url = f"/projects/{project_id}/documents/{document_id}"
    public_url = f"/public/projects/{project_id}/documents/{document_id}"

    denied = client.get(member_url, headers=outsider_headers)
    assert denied.status_code == 403

    allowed = client.get(public_url)
    assert allowed.status_code == 200


# --- Public media and layout ---
# Tests anonymous access to images, layout, and transcription layers. Does not test export zip.


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


# --- Public artifact downloads ---
# Tests anonymous PDF/XML on public routes; member routes still require auth.


@pytest.mark.integration
def test_anonymous_can_download_published_part_artifacts(client, published_document, owner_headers):
    project_id = published_document["project_id"]
    document_id = published_document["document_id"]
    part_id = published_document["part_id"]
    base = f"/public/projects/{project_id}/documents/{document_id}/parts/{part_id}"

    pdf = client.get(f"{base}/transcription-pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")

    xml = client.get(f"{base}/page-xml")
    assert xml.status_code == 200
    assert xml.headers["content-type"] == "application/xml"
    assert xml.content.startswith(b"<?xml")

    draft_pdf = client.get(
        f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/transcription-pdf"
    )
    assert draft_pdf.status_code == 401
