"""Export state — dirty/clean tracking."""

from tests.conftest import minimal_jpeg_bytes

SEGMENT = {
    "id": "seg-1",
    "number": 1,
    "kind": "rectangle",
    "points": [[10, 10], [90, 10], [90, 40], [20, 40]],
    "paired_text_line_index": 1,
}


def _seed_page(data_root, stem="folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text(
        "line one\nline two\n", encoding="utf-8"
    )


def test_page_never_exported_is_dirty(client, data_root):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEGMENT], "export_metadata": None})

    pages = client.get("/pages").json()["pages"]
    assert pages[0]["export_dirty"] is True


def test_export_stub_clears_dirty(client, data_root):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEGMENT], "export_metadata": None})

    export = client.post("/pages/folio/export")
    assert export.status_code == 200

    pages = client.get("/pages").json()["pages"]
    assert pages[0]["export_dirty"] is False


def test_edit_after_export_sets_dirty_again(client, data_root):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEGMENT], "export_metadata": None})
    client.post("/pages/folio/export")

    updated = dict(SEGMENT)
    updated["points"] = [[15, 15], [95, 15], [95, 45], [25, 45]]
    client.put("/pages/folio/annotation", json={"segments": [updated], "export_metadata": None})

    pages = client.get("/pages").json()["pages"]
    assert pages[0]["export_dirty"] is True
