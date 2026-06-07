"""Segment export preview."""

from pathlib import Path

from tests.conftest import minimal_jpeg_bytes

SEG1 = {
    "id": "seg-1",
    "number": 1,
    "kind": "rectangle",
    "points": [[10, 10], [90, 10], [90, 40], [10, 40]],
    "paired_text_line_index": 1,
}


def _seed_page(data_root: Path, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes(200, 120))
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text("line one\n", encoding="utf-8")


def test_segment_preview_returns_jpeg(client, data_root):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG1], "export_metadata": None})

    response = client.get("/pages/folio/segments/seg-1/preview")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_segment_preview_works_for_unpaired_segment(client, data_root):
    unpaired = {**SEG1, "paired_text_line_index": None}
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [unpaired], "export_metadata": None})

    response = client.get("/pages/folio/segments/seg-1/preview")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 100


def test_segment_preview_unknown_segment_returns_404(client, data_root):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG1], "export_metadata": None})

    response = client.get("/pages/folio/segments/missing/preview")

    assert response.status_code == 404
