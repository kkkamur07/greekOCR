"""Document and part integration tests — real Postgres (kalamos)."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier

import pytest
from PIL import Image
from sqlalchemy.exc import IntegrityError

from backend.document.api.schemas import MAX_LINE_GEOMETRY_POINTS
from backend.document.infrastructure.orm_models import Document, DocumentPart
from backend.project.infrastructure.orm_models import Project
from infrastructure.db import sync_system_session
from tests.fixtures.paths import MINIMAL_PNG
from tests.nomicous.integration.helpers import (
    assert_api_error,
    documents_url,
    stored_minimal_page_bytes,
)


def _sized_png_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _create_document_with_part(client, owner_headers, owner_project) -> tuple[str, str, str]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Codex with page"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    return project_id, document_id, upload.json()["id"]


# --- Document CRUD ---
# Tests create, list, read, update, delete for members. Does not test parts or lines.


@pytest.mark.integration
def test_member_create_list_read_update_delete_document(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)

    create = client.post(base, headers=owner_headers, json={"name": "Codex A"})
    assert create.status_code == 201
    doc = create.json()
    document_id = doc["id"]
    assert doc["name"] == "Codex A"
    assert doc["workflow"] == "draft"
    assert doc["part_count"] == 0

    listed = client.get(base, headers=owner_headers)
    assert listed.status_code == 200
    listed_doc = next(d for d in listed.json()["items"] if d["id"] == document_id)
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


# --- Archived workflow ---
# Tests archived documents are hidden unless requested. Does not test public read routes.


@pytest.mark.integration
def test_archived_document_hidden_from_default_list(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)

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
    assert not any(d["id"] == document_id for d in default_list.json()["items"])

    with_archived = client.get(f"{base}?include_archived=true", headers=owner_headers)
    assert with_archived.status_code == 200
    assert any(d["id"] == document_id for d in with_archived.json()["items"])


# --- Document access control ---
# Tests outsiders cannot read or mutate private documents. Does not test collaborators.


@pytest.mark.integration
def test_outsider_cannot_access_documents(client, owner_headers, outsider_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)

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


# --- Parts: upload, reorder, media ---
# Tests part lifecycle and image serving. Does not test line geometry or ML jobs.


@pytest.mark.integration
def test_upload_reorder_delete_part_and_serve_media(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)

    create = client.post(base, headers=owner_headers, json={"name": "With parts"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload_a = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page-a.png", MINIMAL_PNG, "image/png")},
    )
    assert upload_a.status_code == 201
    part_a = upload_a.json()
    assert part_a["order"] == 0
    assert part_a["image_url"] == f"/media/parts/{part_a['id']}"

    upload_b = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page-b.png", MINIMAL_PNG, "image/png")},
    )
    assert upload_b.status_code == 201
    part_b = upload_b.json()
    assert part_b["order"] == 1

    listed = client.get(base, headers=owner_headers)
    listed_doc = next(d for d in listed.json()["items"] if d["id"] == document_id)
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
    assert media.content == stored_minimal_page_bytes()

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


# --- Part upload validation ---
# Tests non-image uploads and reorder validation. Does not test virus scanning.


@pytest.mark.integration
def test_upload_part_rejects_non_image_bytes(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Invalid upload"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    response = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("not-image.txt", b"<script>alert(1)</script>", "text/plain")},
    )

    assert response.status_code == 422
    assert_api_error(
        response,
        code="VALIDATION_ERROR",
        message="Uploaded file is not a valid image",
    )


@pytest.mark.integration
def test_reorder_rejects_duplicate_part_ids(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Reorder dup"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
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
def test_uploaded_part_persists_page_dimensions(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Sized pages"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", _sized_png_bytes(37, 19), "image/png")},
    )
    assert upload.status_code == 201
    assert upload.json()["width"] == 37
    assert upload.json()["height"] == 19

    reread = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert reread.status_code == 200
    part = reread.json()["parts"][0]
    assert (part["width"], part["height"]) == (37, 19)

    with sync_system_session() as session:
        stored = session.get(DocumentPart, uuid.UUID(part["id"]))
        assert (stored.width, stored.height) == (37, 19)


@pytest.mark.integration
def test_reorder_after_deleting_the_leading_parts(client, owner_headers, owner_project):
    """Surviving parts no longer start at order 0; the reorder must not self-collide."""
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Trimmed codex"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    part_ids = []
    for index in range(6):
        upload = client.post(
            f"{base}/{document_id}/parts",
            headers=owner_headers,
            files={"file": (f"page-{index}.png", MINIMAL_PNG, "image/png")},
        )
        assert upload.status_code == 201
        part_ids.append(upload.json()["id"])

    for part_id in part_ids[:5]:
        deleted = client.delete(f"{base}/{document_id}/parts/{part_id}", headers=owner_headers)
        assert deleted.status_code == 204

    for index in range(2):
        upload = client.post(
            f"{base}/{document_id}/parts",
            headers=owner_headers,
            files={"file": (f"extra-{index}.png", MINIMAL_PNG, "image/png")},
        )
        assert upload.status_code == 201
        part_ids.append(upload.json()["id"])

    remaining = [part_ids[5], part_ids[6], part_ids[7]]
    listed = client.get(f"{base}/{document_id}", headers=owner_headers)
    assert [part["order"] for part in listed.json()["parts"]] == [5, 6, 7]

    reorder = client.patch(
        f"{base}/{document_id}/parts/reorder",
        headers=owner_headers,
        json={"part_ids": list(reversed(remaining))},
    )
    assert reorder.status_code == 200
    assert [part["id"] for part in reorder.json()] == list(reversed(remaining))
    assert [part["order"] for part in reorder.json()] == [0, 1, 2]


@pytest.mark.integration
def test_patch_line_clears_block_id_with_an_explicit_null(
    client, owner_headers, owner_project
) -> None:
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    lines_url = f"{documents_url(project_id)}/{document_id}/parts/{part_id}"

    block = client.post(
        f"{lines_url}/blocks",
        headers=owner_headers,
        json={"order": 0, "box": {"x": 0, "y": 0, "width": 10, "height": 10}},
    )
    assert block.status_code == 201
    block_id = block.json()["id"]

    line = client.post(
        f"{lines_url}/lines",
        headers=owner_headers,
        json={"order": 0, "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "block_id": block_id},
    )
    assert line.status_code == 201
    line_id = line.json()["id"]
    assert line.json()["block_id"] == block_id

    cleared = client.patch(
        f"{lines_url}/lines/{line_id}",
        headers=owner_headers,
        json={"block_id": None},
    )
    assert cleared.status_code == 200
    assert cleared.json()["block_id"] is None

    untouched = client.patch(
        f"{lines_url}/lines/{line_id}",
        headers=owner_headers,
        json={"order": 3},
    )
    assert untouched.status_code == 200
    assert untouched.json()["order"] == 3

    rejected = client.patch(
        f"{lines_url}/lines/{line_id}",
        headers=owner_headers,
        json={"order": None},
    )
    assert rejected.status_code == 422


@pytest.mark.integration
def test_line_routes_reject_unbounded_geometry(client, owner_headers, owner_project) -> None:
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    lines_url = f"{documents_url(project_id)}/{document_id}/parts/{part_id}/lines"
    oversized = [[float(index), float(index)] for index in range(MAX_LINE_GEOMETRY_POINTS + 1)]

    response = client.post(
        lines_url,
        headers=owner_headers,
        json={"order": 0, "points": oversized},
    )
    assert response.status_code == 422


@pytest.mark.integration
def test_simultaneous_part_inserts_cannot_duplicate_document_order():
    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Project(id=project_id, name="Concurrent", slug=f"concurrent-{uuid.uuid4().hex}")
        )
        session.add(Document(id=document_id, project_id=project_id, name="Concurrent pages"))
        session.commit()

    barrier = Barrier(2)

    def insert_part() -> bool:
        try:
            with sync_system_session() as session:
                session.add(
                    DocumentPart(
                        document_id=document_id,
                        order=0,
                        image_key=f"parts/{uuid.uuid4()}.webp",
                    )
                )
                barrier.wait(timeout=5)
                session.commit()
            return True
        except IntegrityError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _unused: insert_part(), range(2)))

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1


@pytest.mark.integration
def test_patch_null_name_returns_422(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Named"})
    assert create.status_code == 201
    document_id = create.json()["id"]

    response = client.patch(
        f"{base}/{document_id}",
        headers=owner_headers,
        json={"name": None},
    )
    assert response.status_code == 422


# --- Transcription layers and review ---
# Tests default ground-truth layer and part reviewed toggle. Does not test pairing workflow.


@pytest.mark.integration
def test_document_gets_canonical_ground_truth_transcription_layer(
    client, owner_headers, owner_project
) -> None:
    project_id = owner_project["id"]
    base = documents_url(project_id)

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
    base = documents_url(project_id)

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


# --- Line geometry persistence ---
# Tests replace lines stores baselines, sources, and approved text. Does not run segmentation.


@pytest.mark.integration
def test_replace_part_lines_persists_segment_geometry_and_approved_text(
    client, owner_headers, owner_project
):
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
                    "approved_text": "approved ground truth",
                },
                {
                    "order": 1,
                    "kind": "rectangle",
                    "points": [[0, 10], [10, 10], [10, 15], [0, 15]],
                    "source": "kraken",
                    "source_metadata": {"model": "blla"},
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


# --- Kraken baseline preservation ---
# Tests existing kraken baselines survive point updates. Does not call the inference service.


@pytest.mark.integration
def test_replace_part_lines_preserves_existing_kraken_baseline(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
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
def test_replace_part_lines_preserves_kraken_metadata_when_omitted(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)
    kraken_ceiling = [[-1, 9], [11, 9], [11, 16], [-1, 16]]
    source_metadata = {"model": "blla", "version": "1.0"}
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
                    "source_metadata": source_metadata,
                    "kraken_ceiling": kraken_ceiling,
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 10], [10, 10], [10, 18], [0, 18]],
                    "source": "manual",
                },
            ]
        },
    )
    assert seed.status_code == 200
    kraken_line_id = seed.json()[0]["id"]
    manual_line_id = seed.json()[1]["id"]

    updated = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "id": kraken_line_id,
                    "order": 0,
                    "kind": "polygon",
                    "points": [[1, 1], [11, 1], [11, 9], [1, 9]],
                    "source": "kraken",
                },
                {
                    "id": manual_line_id,
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 11], [10, 11], [10, 19], [0, 19]],
                    "source": "manual",
                },
            ]
        },
    )
    assert updated.status_code == 200
    lines_by_id = {line["id"]: line for line in updated.json()}
    assert lines_by_id[kraken_line_id]["source_metadata"] == source_metadata
    assert lines_by_id[kraken_line_id]["kraken_ceiling"] == kraken_ceiling


# --- Line ID validation ---
# Tests clients cannot supply ids for new lines. Does not test update of existing line ids.


@pytest.mark.integration
def test_replace_part_lines_rejects_client_selected_id_for_new_line(
    client, owner_headers, owner_project
):
    project_id, document_id, part_id = _create_document_with_part(
        client, owner_headers, owner_project
    )
    base = documents_url(project_id)

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


# --- OpenAPI contract ---
# Tests line/transcription schemas are published. Does not validate every endpoint.


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
