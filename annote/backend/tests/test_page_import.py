"""Page import API — upload images into manuscripts/pages."""

from io import BytesIO

from tests.conftest import minimal_jpeg_bytes


def test_import_page_saves_image(client, data_root):
    """POST /pages/import stores the JPEG under manuscripts/pages."""
    jpeg = minimal_jpeg_bytes()
    response = client.post(
        "/pages/import",
        files={"image": ("folio_one.jpg", jpeg, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stem"] == "folio_one"
    assert (data_root / "manuscripts" / "pages" / "folio_one.jpg").is_file()


def test_import_page_with_transcription(client, data_root):
    """Optional transcription file is saved alongside the page stem."""
    jpeg = minimal_jpeg_bytes()
    response = client.post(
        "/pages/import",
        files={
            "image": ("scan.jpg", jpeg, "image/jpeg"),
            "transcription": ("scan.txt", b"line one\nline two", "text/plain"),
        },
    )

    assert response.status_code == 200
    assert response.json()["has_transcription"] is True
    tx = data_root / "transcriptions" / "pages" / "scan.txt"
    assert tx.read_text(encoding="utf-8") == "line one\nline two"


def test_import_page_rejects_non_utf8_transcription(client):
    """Non-UTF-8 transcription uploads return 400."""
    jpeg = minimal_jpeg_bytes()
    response = client.post(
        "/pages/import",
        files={
            "image": ("scan.jpg", jpeg, "image/jpeg"),
            "transcription": ("scan.txt", b"\xff\xfe", "text/plain"),
        },
    )

    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_import_page_rejects_empty_image(client):
    """Empty uploads return 400."""
    response = client.post(
        "/pages/import",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
