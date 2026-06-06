"""Kraken auto-segmentation."""

from types import SimpleNamespace

import pytest

from annote.schemas.annotation import Segment
from annote.services.kraken_segment import (
    auto_segment_page,
    kraken_lines_to_segments,
    pair_segments_to_transcription,
)
from tests.conftest import minimal_jpeg_bytes


def _fake_line(boundary):
    return SimpleNamespace(boundary=boundary)


def test_kraken_lines_to_segments_skips_degenerate():
    lines = [
        _fake_line([[0, 0], [10, 0], [10, 5], [0, 5]]),
        _fake_line([[0, 0], [1, 0]]),
        _fake_line([]),
    ]
    segments = kraken_lines_to_segments(lines, start_number=3)
    assert len(segments) == 1
    assert segments[0].number == 3
    assert segments[0].kind == "polygon"
    assert segments[0].id.startswith("seg-")


def test_pair_segments_to_transcription():
    segments = [
        Segment(id="a", number=1, kind="polygon", points=[[0, 0], [1, 0], [1, 1], [0, 1]]),
        Segment(id="b", number=2, kind="polygon", points=[[0, 0], [1, 0], [1, 1], [0, 1]]),
    ]
    text_lines = [SimpleNamespace(index=1, text="alpha"), SimpleNamespace(index=2, text="beta")]
    paired = pair_segments_to_transcription(segments, text_lines)
    assert paired[0].paired_text_line_index == 1
    assert paired[1].paired_text_line_index == 2


def test_auto_segment_page_replace(client, data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text("line one\nline two\n", encoding="utf-8")
    (data_root / "annotations" / "pages" / f"{stem}.json").write_text(
        '{"segments": [{"id": "old", "number": 1, "kind": "rectangle", '
        '"points": [[0,0],[1,0],[1,1],[0,1]], "paired_text_line_index": null}], "export_metadata": null}',
        encoding="utf-8",
    )

    fake_segments = [
        Segment(id="seg-aaa", number=1, kind="polygon", points=[[1, 1], [9, 1], [9, 4], [1, 4]]),
        Segment(id="seg-bbb", number=2, kind="polygon", points=[[1, 6], [9, 6], [9, 9], [1, 9]]),
    ]

    def fake_segment_image(image, *, device="cpu"):
        assert image.size[0] > 0
        return fake_segments

    monkeypatch.setattr("annote.services.kraken_segment.segment_image", fake_segment_image)

    response = client.post(f"/pages/{stem}/segment", json={"replace": True, "pair_transcription": True})
    assert response.status_code == 200
    body = response.json()
    assert len(body["segments"]) == 2
    assert body["segments"][0]["paired_text_line_index"] == 1
    assert body["segments"][1]["paired_text_line_index"] == 2
    assert all(s["id"].startswith("seg-") for s in body["segments"])


def test_auto_segment_page_append(client, data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())
    (data_root / "annotations" / "pages" / f"{stem}.json").write_text(
        '{"segments": [{"id": "keep", "number": 2, "kind": "rectangle", '
        '"points": [[0,0],[1,0],[1,1],[0,1]], "paired_text_line_index": null}], "export_metadata": null}',
        encoding="utf-8",
    )

    fake_segments = [
        Segment(id="seg-new", number=1, kind="polygon", points=[[1, 1], [9, 1], [9, 4], [1, 4]]),
    ]

    monkeypatch.setattr(
        "annote.services.kraken_segment.segment_image",
        lambda image, *, device="cpu": fake_segments,
    )

    response = client.post(f"/pages/{stem}/segment", json={"replace": False, "pair_transcription": False})
    assert response.status_code == 200
    segments = response.json()["segments"]
    assert len(segments) == 2
    assert segments[0]["id"] == "keep"
    assert segments[1]["number"] == 3


def test_auto_segment_requires_kraken_message(client, data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())

    def boom(*_args, **_kwargs):
        raise RuntimeError("Kraken is required for auto-segmentation. Install with: pip install 'annote[kraken]'")

    monkeypatch.setattr("annote.services.kraken_segment.auto_segment_page", boom)

    response = client.post(f"/pages/{stem}/segment", json={})
    assert response.status_code == 400
    assert "Kraken is required" in response.json()["detail"]


def test_segment_image_on_real_page():
    pytest.importorskip("kraken")
    from PIL import Image

    from annote.services.kraken_segment import segment_image

    image = Image.new("RGB", (400, 120), "white")
    for y in range(20, 100, 30):
        for x in range(40, 360, 8):
            image.putpixel((x, y), (0, 0, 0))

    segments = segment_image(image)
    assert len(segments) >= 1
    assert all(len(s.points) >= 3 for s in segments)
