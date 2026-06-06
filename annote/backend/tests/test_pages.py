"""Page catalogue API — list and serve page images."""

from tests.conftest import minimal_jpeg_bytes


def test_list_pages_empty(client):
    """Empty pages directory returns an empty list."""
    response = client.get("/pages")
    assert response.status_code == 200
    assert response.json() == {"pages": []}


def test_list_pages_finds_fixture_file(client, data_root):
    """list_pages discovers JPEG files sorted by filename."""
    pages_dir = data_root / "manuscripts" / "pages"
    (pages_dir / "b_page.jpg").write_bytes(minimal_jpeg_bytes())
    (pages_dir / "a_page.jpg").write_bytes(minimal_jpeg_bytes())

    response = client.get("/pages")

    assert response.status_code == 200
    stems = [p["stem"] for p in response.json()["pages"]]
    assert stems == ["a_page", "b_page"]


def test_get_page_image_returns_bytes(client, data_root):
    """GET /pages/{stem}/image serves the page JPEG."""
    jpeg = minimal_jpeg_bytes()
    (data_root / "manuscripts" / "pages" / "folio.jpg").write_bytes(jpeg)

    response = client.get("/pages/folio/image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == jpeg


def test_get_page_image_unknown_stem_returns_404(client):
    """Unknown page stem returns 404."""
    response = client.get("/pages/missing/image")
    assert response.status_code == 404
