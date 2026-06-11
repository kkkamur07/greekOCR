"""OCR prediction — Calamari pairing assist."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from annote.schemas.annotation import PageAnnotation, Segment
from annote.services.annotation_merge import clear_stale_model_transcriptions
from annote.services.calamari_ocr import _checkpoint_exists, predict_segment_text
from tests.conftest import minimal_jpeg_bytes

SEG = {
    "id": "seg-1",
    "number": 1,
    "kind": "rectangle",
    "points": [[10, 10], [90, 10], [90, 40], [10, 40]],
    "paired_text_line_index": None,
}


def _seed_page(data_root, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes(200, 120))


def test_checkpoint_exists_accepts_savedmodel_directory(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.mkdir()
    (ckpt / "saved_model.pb").write_bytes(b"model")
    (tmp_path / "best.ckpt.json").write_text("{}")
    assert _checkpoint_exists(ckpt) is True


def test_checkpoint_exists_accepts_metadata_path(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.mkdir()
    (ckpt / "saved_model.pb").write_bytes(b"model")
    metadata = tmp_path / "best.ckpt.json"
    metadata.write_text("{}")
    assert _checkpoint_exists(metadata) is True


def test_checkpoint_exists_rejects_missing_path(tmp_path):
    assert _checkpoint_exists(tmp_path / "missing.ckpt") is False


def test_checkpoint_exists_rejects_model_without_json(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.mkdir()
    (ckpt / "saved_model.pb").write_bytes(b"model")
    assert _checkpoint_exists(ckpt) is False


def test_segment_defaults_model_transcription_null():
    segment = Segment(
        id="seg-1",
        number=1,
        kind="polygon",
        points=[[0, 0], [10, 0], [10, 5], [0, 5]],
    )
    assert segment.model_transcription is None
    assert segment.model_transcription_at is None


def test_page_annotation_accepts_legacy_segments_without_model_fields():
    annotation = PageAnnotation.model_validate(
        {
            "segments": [
                {
                    "id": "seg-1",
                    "number": 1,
                    "kind": "polygon",
                    "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "paired_text_line_index": None,
                }
            ],
            "export_metadata": None,
        }
    )
    assert annotation.segments[0].model_transcription is None
    assert annotation.segments[0].model_transcription_at is None


def test_clear_stale_model_transcription_when_points_change():
    existing = PageAnnotation(
        segments=[
            Segment(
                id="seg-1",
                number=1,
                kind="rectangle",
                points=[[0, 0], [10, 0], [10, 5], [0, 5]],
                model_transcription="old text",
                model_transcription_at="2026-01-01T00:00:00Z",
            )
        ]
    )
    incoming = PageAnnotation(
        segments=[
            Segment(
                id="seg-1",
                number=1,
                kind="rectangle",
                points=[[1, 0], [10, 0], [10, 5], [0, 5]],
                model_transcription="old text",
                model_transcription_at="2026-01-01T00:00:00Z",
            )
        ]
    )

    merged = clear_stale_model_transcriptions(existing, incoming)
    assert merged.segments[0].model_transcription is None
    assert merged.segments[0].model_transcription_at is None


def test_clear_stale_model_transcription_preserves_when_points_unchanged():
    existing = PageAnnotation(
        segments=[
            Segment(
                id="seg-1",
                number=1,
                kind="rectangle",
                points=[[0, 0], [10, 0], [10, 5], [0, 5]],
                model_transcription="kept",
                model_transcription_at="2026-01-01T00:00:00Z",
            )
        ]
    )
    incoming = existing.model_copy(deep=True)
    incoming.segments[0].paired_text_line_index = 1

    merged = clear_stale_model_transcriptions(existing, incoming)
    assert merged.segments[0].model_transcription == "kept"
    assert merged.segments[0].model_transcription_at == "2026-01-01T00:00:00Z"


def test_predict_segment_text_uses_rectify_and_predictor(monkeypatch):
    page = np.zeros((50, 100, 3), dtype=np.uint8)
    segment = Segment(id="seg-1", number=1, kind="rectangle", points=[[0, 0], [10, 0], [10, 5], [0, 5]])
    rectify_called = False

    def fake_rectify(image, seg):
        nonlocal rectify_called
        rectify_called = True
        return np.ones((5, 10, 3), dtype=np.uint8) * 128

    monkeypatch.setattr("annote.services.calamari_ocr.rectify", fake_rectify)

    class FakePredictor:
        def predict_raw(self, images):
            assert len(images) == 1
            assert images[0].ndim == 2
            assert images[0].dtype == np.uint8
            return [SimpleNamespace(outputs=SimpleNamespace(sentence=SimpleNamespace(sentence="predicted line")))]

    monkeypatch.setattr("annote.services.calamari_ocr._get_predictor", lambda: FakePredictor())

    text = predict_segment_text(page, segment)
    assert text == "predicted line"
    assert rectify_called is True


def test_post_segment_ocr_returns_model_fields(client, data_root, monkeypatch):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG], "export_metadata": None})

    monkeypatch.setattr(
        "annote.services.calamari_ocr.predict_segment_text",
        lambda _page, _seg: "model line",
    )

    response = client.post("/pages/folio/segments/seg-1/ocr")
    assert response.status_code == 200
    body = response.json()
    seg = body["segments"][0]
    assert seg["model_transcription"] == "model line"
    assert seg["model_transcription_at"] is not None


def test_post_segment_ocr_404_for_unknown_segment(client, data_root, monkeypatch):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG], "export_metadata": None})
    monkeypatch.setattr("annote.services.calamari_ocr.predict_segment_text", lambda *_: "x")

    response = client.post("/pages/folio/segments/missing/ocr")
    assert response.status_code == 404


def test_post_segment_ocr_allowed_when_locked(client, data_root, monkeypatch):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG], "export_metadata": None})
    client.post("/pages/folio/lock")
    monkeypatch.setattr(
        "annote.services.calamari_ocr.predict_segment_text",
        lambda _page, _seg: "locked ocr",
    )

    response = client.post("/pages/folio/segments/seg-1/ocr")
    assert response.status_code == 200
    assert response.json()["segments"][0]["model_transcription"] == "locked ocr"


def test_put_annotation_clears_model_fields_when_points_change(client, data_root):
    _seed_page(data_root)
    seg = {**SEG, "model_transcription": "stale", "model_transcription_at": "2026-01-01T00:00:00Z"}
    client.put("/pages/folio/annotation", json={"segments": [seg], "export_metadata": None})

    moved = dict(seg)
    moved["points"] = [[11, 10], [90, 10], [90, 40], [10, 40]]
    client.put("/pages/folio/annotation", json={"segments": [moved], "export_metadata": None})

    get = client.get("/pages/folio/annotation")
    saved = get.json()["segments"][0]
    assert saved["model_transcription"] is None
    assert saved["model_transcription_at"] is None


def test_calamari_missing_returns_clear_error(client, data_root, monkeypatch):
    _seed_page(data_root)
    client.put("/pages/folio/annotation", json={"segments": [SEG], "export_metadata": None})

    def boom(*_args, **_kwargs):
        raise RuntimeError("Calamari is required for OCR. Install with: pip install -e '.[calamari]'")

    monkeypatch.setattr("annote.services.calamari_ocr.predict_segment_text", boom)

    response = client.post("/pages/folio/segments/seg-1/ocr")
    assert response.status_code == 400
    assert "Calamari is required" in response.json()["detail"]


def test_page_ocr_stream_emits_progress_and_updates_all_segments(client, data_root, monkeypatch):
    _seed_page(data_root)
    seg2 = {**SEG, "id": "seg-2", "number": 2, "points": [[10, 50], [90, 50], [90, 80], [10, 80]]}
    client.put("/pages/folio/annotation", json={"segments": [SEG, seg2], "export_metadata": None})

    call_count = 0

    def fake_predict(_page, _seg):
        nonlocal call_count
        call_count += 1
        return f"line-{call_count}"

    monkeypatch.setattr("annote.services.calamari_ocr.predict_segment_text", fake_predict)

    response = client.post("/pages/folio/ocr/stream")
    assert response.status_code == 200

    events = [json.loads(line) for line in response.text.strip().split("\n") if line]
    progress = [e for e in events if e["type"] == "progress"]
    done = [e for e in events if e["type"] == "done"]

    assert len(progress) == 2
    assert progress[0]["current"] == 1 and progress[0]["total"] == 2
    assert progress[1]["current"] == 2
    assert done[0]["result"]["processed_count"] == 2

    ann = client.get("/pages/folio/annotation").json()
    assert ann["segments"][0]["model_transcription"] == "line-1"
    assert ann["segments"][1]["model_transcription"] == "line-2"


def test_page_ocr_stream_zero_segments_completes(client, data_root):
    _seed_page(data_root)

    response = client.post("/pages/folio/ocr/stream")
    assert response.status_code == 200

    events = [json.loads(line) for line in response.text.strip().split("\n") if line]
    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert events[0]["result"]["processed_count"] == 0
