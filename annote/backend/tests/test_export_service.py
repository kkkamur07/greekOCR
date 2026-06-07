"""Export service — paired segments to line files."""

from pathlib import Path

from PIL import Image

from annote.services.export_service import export_page_events
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
UNPAIRED = {
    "id": "seg-3",
    "number": 3,
    "kind": "rectangle",
    "points": [[10, 85], [90, 85], [90, 95], [10, 95]],
    "paired_text_line_index": None,
}


def _seed_page(data_root: Path, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes(200, 120))
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text(
        "αἱρετικῶν\nκαὶ φιλοσόφων\nunused line\n", encoding="utf-8"
    )


def test_export_writes_paired_jpg_and_txt_only(client, data_root):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2, UNPAIRED], "export_metadata": None},
    )

    response = client.post("/pages/folio/export")
    assert response.status_code == 200
    body = response.json()
    assert body["exported_count"] == 2
    assert len(body["warnings"]["unpaired_segments"]) == 1
    assert body["warnings"]["unused_text_lines"] == [3]

    export = data_root / "manuscripts" / "export"
    assert (export / "folio_1.jpg").is_file()
    assert (export / "folio_2.jpg").is_file()
    assert not (export / "folio_3.jpg").exists()

    txt1 = (export / "folio_1.txt").read_text(encoding="utf-8")
    assert txt1 == "αἱρετικῶν"
    txt2 = (export / "folio_2.txt").read_text(encoding="utf-8")
    assert txt2 == "καὶ φιλοσόφων"


def test_export_empty_text_override_without_pairing(client, data_root):
    _seed_page(data_root)
    seg = {
        **SEG1,
        "paired_text_line_index": None,
        "text_override": "",
    }
    client.put("/pages/folio/annotation", json={"segments": [seg], "export_metadata": None})

    response = client.post("/pages/folio/export")
    assert response.status_code == 200
    assert response.json()["exported_count"] == 1

    txt = (data_root / "manuscripts" / "export" / "folio_1.txt").read_text(encoding="utf-8")
    assert txt == ""


def test_export_text_override_without_pairing(client, data_root):
    _seed_page(data_root)
    seg = {
        **SEG1,
        "paired_text_line_index": None,
        "text_override": "typed directly",
    }
    client.put("/pages/folio/annotation", json={"segments": [seg], "export_metadata": None})

    response = client.post("/pages/folio/export")
    assert response.status_code == 200
    assert response.json()["exported_count"] == 1

    txt = (data_root / "manuscripts" / "export" / "folio_1.txt").read_text(encoding="utf-8")
    assert txt == "typed directly"


def test_reexport_overwrites_line_files(client, data_root):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG1], "export_metadata": None})
    client.post("/pages/folio/export")

    seg = dict(SEG1)
    seg["paired_text_line_index"] = 2
    client.put("/pages/folio/annotation", json={"segments": [seg], "export_metadata": None})
    client.post("/pages/folio/export")

    txt = (data_root / "manuscripts" / "export" / "folio_1.txt").read_text(encoding="utf-8")
    assert txt == "καὶ φιλοσόφων"


def test_export_uses_maximum_jpeg_quality(client, data_root, monkeypatch):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG1], "export_metadata": None})

    saved_kwargs: dict = {}
    original_save = Image.Image.save

    def capture_save(self, fp, format=None, **kwargs):
        if format == "JPEG":
            saved_kwargs.update(kwargs)
        return original_save(self, fp, format=format, **kwargs)

    monkeypatch.setattr(Image.Image, "save", capture_save)

    response = client.post("/pages/folio/export")
    assert response.status_code == 200
    assert saved_kwargs["quality"] == 100
    assert saved_kwargs["subsampling"] == 0


def test_export_stream_reports_progress(client, data_root):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None},
    )

    response = client.post("/pages/folio/export/stream")
    assert response.status_code == 200

    events = [__import__("json").loads(line) for line in response.text.strip().split("\n") if line]
    progress = [e for e in events if e["type"] == "progress"]
    done = [e for e in events if e["type"] == "done"]

    assert len(done) == 1
    assert done[0]["result"]["exported_count"] == 2
    assert any(e["step"] == "rectify" for e in progress)
    assert progress[-1]["step"] == "save"


def test_export_progress_yields_rectify_before_running_it(client, data_root, monkeypatch):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG1], "export_metadata": None})
    rectify_called = False

    def fake_rectify(image, segment):
        nonlocal rectify_called
        rectify_called = True
        return image

    monkeypatch.setattr("annote.services.processing.pipeline.rectify", fake_rectify)

    events = export_page_events(data_root, "folio")
    first = next(events)

    assert first.step == "rectify"
    assert rectify_called is False

    list(events)
    assert rectify_called is True
