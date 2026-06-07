"""Transcription PDF share mode — frozen PDF at page lock."""

import io

from pypdf import PdfReader

from tests.conftest import minimal_jpeg_bytes

SEG1 = {
    "id": "seg-1",
    "number": 1,
    "kind": "rectangle",
    "points": [[10, 10], [90, 10], [90, 40], [10, 40]],
    "paired_text_line_index": 1,
}
SEG2 = {
    "id": "seg-2",
    "number": 2,
    "kind": "rectangle",
    "points": [[10, 50], [90, 50], [90, 80], [10, 80]],
    "paired_text_line_index": 2,
}


def _seed_page(data_root, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes(200, 120))
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text(
        "αἱρετικῶν\nκαὶ φιλοσόφων\n",
        encoding="utf-8",
    )


def _pdf_text(pdf_bytes: bytes) -> str:
    return PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text() or ""


def test_lock_writes_share_pdf(client, data_root, unicode_font):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None, "locked": False},
    )

    client.post("/pages/folio/lock")

    share_path = data_root / "manuscripts" / "share" / "folio_transcription.pdf"
    assert share_path.is_file()
    assert share_path.read_bytes().startswith(b"%PDF")


def test_share_pdf_endpoint_404_when_unlocked(client, data_root, unicode_font):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None, "locked": False},
    )
    client.post("/pages/folio/lock")
    client.post("/pages/folio/unlock")

    response = client.get("/pages/folio/transcription.share.pdf")

    assert response.status_code == 404


def test_share_pdf_served_when_locked(client, data_root, unicode_font):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None, "locked": False},
    )
    client.post("/pages/folio/lock")

    response = client.get("/pages/folio/transcription.share.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "αἱρετικῶν" in _pdf_text(response.content)


def test_preview_pdf_reflects_edits_after_unlock(client, data_root, unicode_font):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None, "locked": False},
    )
    client.post("/pages/folio/lock")
    share_at_lock = _pdf_text(client.get("/pages/folio/transcription.share.pdf").content)

    client.post("/pages/folio/unlock")
    seg2_edited = {**SEG2, "text_override": "edited override text"}
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, seg2_edited], "export_metadata": None, "locked": False},
    )

    preview = _pdf_text(client.get("/pages/folio/transcription.pdf").content)
    assert "edited override text" in preview
    assert share_at_lock != preview or "edited override text" not in share_at_lock


def test_lock_rolls_back_when_share_pdf_write_fails(client, data_root, unicode_font, monkeypatch):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None, "locked": False},
    )

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "annote.api.pages.write_share_pdf_bytes",
        fail_write,
    )

    response = client.post("/pages/folio/lock")

    assert response.status_code == 500
    assert client.get("/pages/folio/annotation").json()["locked"] is False
    assert not (data_root / "manuscripts" / "share" / "folio_transcription.pdf").exists()
