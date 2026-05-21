"""Project CRUD and sharing integration tests — real Postgres (kalamos)."""

import uuid

import pytest


@pytest.mark.integration
def test_list_projects_empty_for_new_user(client, auth_headers):
    response = client.get("/projects", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.integration
def test_owner_create_read_update_delete(client, owner_headers):
    slug = f"proj-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "My Manuscripts", "guidelines": "Use polytonic Greek"},
    )
    assert create.status_code == 201
    body = create.json()
    project_id = body["id"]
    assert body["slug"] == slug
    assert body["name"] == "My Manuscripts"
    assert body["guidelines"] == "Use polytonic Greek"
    assert "owner_id" in body

    read = client.get(f"/projects/{project_id}", headers=owner_headers)
    assert read.status_code == 200
    assert read.json()["slug"] == slug

    update = client.patch(
        f"/projects/{project_id}",
        headers=owner_headers,
        json={"name": "Renamed", "guidelines": None},
    )
    assert update.status_code == 200
    assert update.json()["name"] == "Renamed"
    assert update.json()["guidelines"] is None

    listed = client.get("/projects", headers=owner_headers)
    assert listed.status_code == 200
    ids = [p["id"] for p in listed.json()]
    assert project_id in ids

    delete = client.delete(f"/projects/{project_id}", headers=owner_headers)
    assert delete.status_code == 204

    gone = client.get(f"/projects/{project_id}", headers=owner_headers)
    assert gone.status_code == 404


@pytest.mark.integration
def test_create_duplicate_slug_conflict(client, owner_headers):
    slug = f"dup-{uuid.uuid4().hex[:8]}"
    first = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "First"},
    )
    assert first.status_code == 201
    second = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Second"},
    )
    assert second.status_code == 409


@pytest.mark.integration
def test_share_and_collaborator_list_read(client, owner_headers, collaborator_headers, collaborator_user):
    slug = f"share-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Shared workspace"},
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    share = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    assert share.status_code == 204

    collab_list = client.get("/projects", headers=collaborator_headers)
    assert collab_list.status_code == 200
    assert any(p["id"] == project_id for p in collab_list.json())

    collab_read = client.get(f"/projects/{project_id}", headers=collaborator_headers)
    assert collab_read.status_code == 200
    assert collab_read.json()["slug"] == slug

    unshare = client.delete(
        f"/projects/{project_id}/share/{collaborator_user['username']}",
        headers=owner_headers,
    )
    assert unshare.status_code == 204

    collab_list_after = client.get("/projects", headers=collaborator_headers)
    assert not any(p["id"] == project_id for p in collab_list_after.json())

    collab_read_after = client.get(f"/projects/{project_id}", headers=collaborator_headers)
    assert collab_read_after.status_code in (403, 404)


@pytest.mark.integration
def test_non_member_cannot_read_or_mutate(
    client, owner_headers, outsider_headers, outsider_user
):
    slug = f"private-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Private"},
    )
    project_id = create.json()["id"]

    read = client.get(f"/projects/{project_id}", headers=outsider_headers)
    assert read.status_code in (403, 404)

    patch = client.patch(
        f"/projects/{project_id}",
        headers=outsider_headers,
        json={"name": "Hijacked"},
    )
    assert patch.status_code in (403, 404)

    delete = client.delete(f"/projects/{project_id}", headers=outsider_headers)
    assert delete.status_code in (403, 404)

    share = client.post(
        f"/projects/{project_id}/share",
        headers=outsider_headers,
        json={"username": outsider_user["username"]},
    )
    assert share.status_code in (403, 404)


@pytest.mark.integration
def test_collaborator_cannot_update_or_delete(
    client, owner_headers, collaborator_headers, collaborator_user
):
    slug = f"collab-{uuid.uuid4().hex[:8]}"
    create = client.post(
        "/projects",
        headers=owner_headers,
        json={"slug": slug, "name": "Team project"},
    )
    project_id = create.json()["id"]
    client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )

    patch = client.patch(
        f"/projects/{project_id}",
        headers=collaborator_headers,
        json={"name": "Not allowed"},
    )
    assert patch.status_code == 403

    delete = client.delete(f"/projects/{project_id}", headers=collaborator_headers)
    assert delete.status_code == 403
