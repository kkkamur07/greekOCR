"""Integration coverage for part-media variants and HTTP cache behavior."""

from io import BytesIO

import pytest
from PIL import Image

from backend.document.infrastructure.media_store.encoding import encode_part_image


def _png_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, color=(80, 120, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _create_part(client, owner_headers, owner_project, image: bytes) -> tuple[str, str]:
    documents_url = f"/projects/{owner_project['id']}/documents"
    document = client.post(documents_url, headers=owner_headers, json={"name": "Media variants"})
    assert document.status_code == 201

    upload = client.post(
        f"{documents_url}/{document.json()['id']}/parts",
        headers=owner_headers,
        files={"file": ("folio.png", image, "image/png")},
    )
    assert upload.status_code == 201
    return document.json()["id"], upload.json()["id"]


@pytest.mark.integration
def test_private_part_media_preserves_full_bytes_and_authorizes_before_cache(
    client, owner_headers, outsider_headers, owner_project
):
    source = _png_bytes((400, 100))
    _, part_id = _create_part(client, owner_headers, owner_project, source)
    url = f"/media/parts/{part_id}"

    full = client.get(url, headers=owner_headers)
    assert full.status_code == 200
    assert full.content == encode_part_image(source)
    assert full.headers["content-type"] == "image/webp"
    assert full.headers["cache-control"] == "private, max-age=86400"

    etag = full.headers["etag"]
    assert client.get(url).status_code == 401
    assert client.get(url, headers=outsider_headers).status_code == 403
    conditional = client.get(url, headers={**owner_headers, "If-None-Match": etag})
    assert conditional.status_code == 304
    assert conditional.headers["etag"] == etag


@pytest.mark.integration
def test_part_thumbnail_is_bounded_lossy_webp_without_upscaling(
    client, owner_headers, owner_project
):
    _, part_id = _create_part(client, owner_headers, owner_project, _png_bytes((400, 100)))
    url = f"/media/parts/{part_id}"

    thumbnail = client.get(f"{url}?w=200", headers=owner_headers)
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/webp"
    with Image.open(BytesIO(thumbnail.content)) as image:
        assert image.format == "WEBP"
        assert image.size == (200, 50)

    non_upscaled = client.get(f"{url}?w=1000", headers=owner_headers)
    assert non_upscaled.status_code == 200
    with Image.open(BytesIO(non_upscaled.content)) as image:
        assert image.size == (400, 100)


@pytest.mark.integration
def test_part_media_width_validation_and_variant_etags(client, owner_headers, owner_project):
    _, part_id = _create_part(client, owner_headers, owner_project, _png_bytes((400, 100)))
    url = f"/media/parts/{part_id}"

    for invalid_width in ("0", "not-a-number", "2049"):
        response = client.get(f"{url}?w={invalid_width}", headers=owner_headers)
        assert response.status_code == 422

    full = client.get(url, headers=owner_headers)
    thumb_200 = client.get(f"{url}?w=200", headers=owner_headers)
    thumb_201 = client.get(f"{url}?w=201", headers=owner_headers)
    assert len({full.headers["etag"], thumb_200.headers["etag"], thumb_201.headers["etag"]}) == 3


@pytest.mark.integration
def test_public_part_media_requires_publication_before_conditional_cache(
    client, owner_headers, owner_project
):
    source = _png_bytes((400, 100))
    document_id, part_id = _create_part(client, owner_headers, owner_project, source)
    public_url = f"/public/media/parts/{part_id}"
    document_url = f"/projects/{owner_project['id']}/documents/{document_id}"

    assert client.get(public_url).status_code == 404
    publish = client.patch(document_url, headers=owner_headers, json={"workflow": "published"})
    assert publish.status_code == 200

    full = client.get(public_url)
    assert full.status_code == 200
    assert full.content == encode_part_image(source)
    assert full.headers["cache-control"] == "public, max-age=300, must-revalidate"
    assert client.get(f"{public_url}?w=0").status_code == 422

    etag = full.headers["etag"]
    assert client.get(public_url, headers={"If-None-Match": etag}).status_code == 304
    unpublish = client.patch(document_url, headers=owner_headers, json={"workflow": "draft"})
    assert unpublish.status_code == 200
    assert client.get(public_url, headers={"If-None-Match": etag}).status_code == 404
