"""Page lock — freeze annotation edits until unlocked."""

from tests.conftest import minimal_jpeg_bytes

SAMPLE_SEGMENT = {
    "id": "seg-1",
    "number": 1,
    "kind": "rectangle",
    "points": [[10, 10], [90, 10], [90, 40], [10, 40]],
    "paired_text_line_index": None,
}


def _seed_page(data_root, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())


def test_lock_page_sets_locked_state(client, data_root):
    _seed_page(data_root)

    response = client.post("/pages/folio/lock")

    assert response.status_code == 200
    assert response.json()["locked"] is True

    get = client.get("/pages/folio/annotation")
    assert get.status_code == 200
    assert get.json()["locked"] is True


def test_unlock_page_clears_locked_state(client, data_root):
    _seed_page(data_root)
    client.post("/pages/folio/lock")

    response = client.post("/pages/folio/unlock")

    assert response.status_code == 200
    assert response.json()["locked"] is False
    assert client.get("/pages/folio/annotation").json()["locked"] is False


def test_put_annotation_rejected_when_locked(client, data_root):
    _seed_page(data_root)
    client.post("/pages/folio/lock")

    response = client.put(
        "/pages/folio/annotation",
        json={"segments": [SAMPLE_SEGMENT], "export_metadata": None, "locked": True},
    )

    assert response.status_code == 409


def test_auto_segment_rejected_when_locked(client, data_root):
    _seed_page(data_root)
    client.post("/pages/folio/lock")

    response = client.post("/pages/folio/segment", json={})

    assert response.status_code == 409


def test_list_pages_exposes_locked(client, data_root):
    _seed_page(data_root)
    client.post("/pages/folio/lock")

    response = client.get("/pages")

    assert response.status_code == 200
    page = response.json()["pages"][0]
    assert page["stem"] == "folio"
    assert page["locked"] is True


def test_lock_unlock_does_not_mark_export_dirty(client, data_root):
    _seed_page(data_root)
    (data_root / "transcriptions" / "pages" / "folio.txt").write_text("line one\n", encoding="utf-8")
    seg = {**SAMPLE_SEGMENT, "paired_text_line_index": 1}
    client.put("/pages/folio/annotation", json={"segments": [seg], "export_metadata": None})
    client.post("/pages/folio/export")
    assert client.get("/pages").json()["pages"][0]["export_dirty"] is False

    client.post("/pages/folio/lock")
    assert client.get("/pages").json()["pages"][0]["export_dirty"] is False

    client.post("/pages/folio/unlock")
    assert client.get("/pages").json()["pages"][0]["export_dirty"] is False


def test_page_lock_settings_env_override(monkeypatch):
    from annote.settings import Settings, get_settings

    monkeypatch.setenv("ANNOTE_PAGE_LOCK_PROMPT_AT_FULL_PAIRING", "false")
    get_settings.cache_clear()
    settings = Settings()
    assert settings.page_lock.prompt_at_full_pairing is False
    get_settings.cache_clear()
