"""Annotation history — snapshots, retention, restore."""

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
    "paired_text_line_index": None,
}


def _seed_page(data_root, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text(
        "line one\nline two\n",
        encoding="utf-8",
    )


def test_history_empty_for_new_page(client, data_root):
    _seed_page(data_root)

    response = client.get("/pages/folio/history")

    assert response.status_code == 200
    assert response.json()["snapshots"] == []


def test_milestone_snapshot_when_pairing_reaches_threshold(client, data_root):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None, "locked": False},
    )

    history = client.get("/pages/folio/history").json()["snapshots"]
    reasons = [s["reason"] for s in history]

    assert "milestone_50" in reasons


def test_milestone_100_when_all_segments_paired(client, data_root):
    _seed_page(data_root)
    seg2_paired = {**SEG2, "paired_text_line_index": 2}
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, seg2_paired], "export_metadata": None, "locked": False},
    )

    reasons = [s["reason"] for s in client.get("/pages/folio/history").json()["snapshots"]]
    assert "milestone_100" in reasons


def test_lock_and_unlock_write_protected_snapshots(client, data_root):
    _seed_page(data_root)
    client.post("/pages/folio/lock")
    client.post("/pages/folio/unlock")

    reasons = [s["reason"] for s in client.get("/pages/folio/history").json()["snapshots"]]
    assert "lock" in reasons
    assert "unlock" in reasons


def test_restore_snapshot_replaces_annotation(client, data_root):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG2], "export_metadata": None, "locked": False},
    )
    snapshots = client.get("/pages/folio/history").json()["snapshots"]
    snapshot_id = snapshots[0]["id"]

    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None, "locked": False},
    )
    assert len(client.get("/pages/folio/annotation").json()["segments"]) == 2

    response = client.post(f"/pages/folio/history/{snapshot_id}/restore")
    assert response.status_code == 200

    restored = client.get("/pages/folio/annotation").json()
    assert len(restored["segments"]) == 1
    assert restored["segments"][0]["id"] == "seg-2"


def test_restore_rejected_when_page_locked(client, data_root):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG2], "export_metadata": None, "locked": False},
    )
    snapshot_id = client.get("/pages/folio/history").json()["snapshots"][0]["id"]
    client.post("/pages/folio/lock")

    response = client.post(f"/pages/folio/history/{snapshot_id}/restore")

    assert response.status_code == 409


def test_timed_snapshot_retention_prunes_oldest(client, data_root, monkeypatch):
    monkeypatch.setenv("ANNOTE_HISTORY_MAX_TIMED_SNAPSHOTS", "2")
    monkeypatch.setenv("ANNOTE_HISTORY_SNAPSHOT_INTERVAL_MINUTES", "0")
    from annote.settings import get_settings

    get_settings.cache_clear()

    _seed_page(data_root)
    for i in range(4):
        seg = {
            "id": f"seg-{i}",
            "number": i + 1,
            "kind": "rectangle",
            "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
            "paired_text_line_index": None,
        }
        client.put(
            "/pages/folio/annotation",
            json={"segments": [seg], "export_metadata": None, "locked": False},
        )

    timed = [s for s in client.get("/pages/folio/history").json()["snapshots"] if s["reason"] == "timed"]
    assert len(timed) <= 2

    get_settings.cache_clear()


def test_restore_clears_export_metadata_and_marks_dirty(client, data_root):
    _seed_page(data_root)
    seg = {**SEG1, "paired_text_line_index": 1}
    client.put(
        "/pages/folio/annotation",
        json={"segments": [seg], "export_metadata": None, "locked": False},
    )
    client.post("/pages/folio/export")
    assert client.get("/pages").json()["pages"][0]["export_dirty"] is False

    seg_edited = {**seg, "text_override": "edited"}
    client.put(
        "/pages/folio/annotation",
        json={"segments": [seg_edited], "export_metadata": None, "locked": False},
    )
    snapshot_id = client.get("/pages/folio/history").json()["snapshots"][0]["id"]

    client.post(f"/pages/folio/history/{snapshot_id}/restore")

    restored = client.get("/pages/folio/annotation").json()
    assert restored["export_metadata"] is None
    assert client.get("/pages").json()["pages"][0]["export_dirty"] is True


def test_auto_segment_writes_history_snapshot(client, data_root, monkeypatch):
    import annote.api.pages as pages_api

    seg = {
        "id": "seg-auto",
        "number": 1,
        "kind": "rectangle",
        "points": [[10, 10], [90, 10], [90, 40], [10, 40]],
        "paired_text_line_index": None,
    }

    def fake_auto_segment_page(data_root, stem, **kwargs):
        from annote.schemas.annotation import PageAnnotation, Segment
        from annote.services.annotation_store import save_annotation

        return save_annotation(
            data_root,
            stem,
            PageAnnotation(segments=[Segment(**seg)]),
        )

    monkeypatch.setattr(pages_api, "auto_segment_page", fake_auto_segment_page)

    _seed_page(data_root)
    client.post("/pages/folio/segment", json={})

    reasons = [s["reason"] for s in client.get("/pages/folio/history").json()["snapshots"]]
    assert "timed" in reasons
