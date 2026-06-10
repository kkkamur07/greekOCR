"""Annotation store — segment geometry and pairings."""

from annote.schemas.annotation import PageAnnotation, Segment

from tests.conftest import minimal_jpeg_bytes

SAMPLE_POLYGON = {
    "id": "seg-1",
    "number": 1,
    "kind": "polygon",
    "points": [[10, 10], [90, 10], [90, 40], [10, 40]],
    "paired_text_line_index": None,
}

SAMPLE_RECTANGLE = {
    "id": "seg-2",
    "number": 2,
    "kind": "rectangle",
    "points": [[20, 50], [80, 50], [80, 70], [20, 70]],
    "paired_text_line_index": None,
}


def _seed_page(data_root, stem="folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())


def test_annotation_round_trip_polygon_and_rectangle(client, data_root):
    _seed_page(data_root)
    payload = {"segments": [SAMPLE_POLYGON, SAMPLE_RECTANGLE], "export_metadata": None}

    put = client.put("/pages/folio/annotation", json=payload)
    assert put.status_code == 200

    get = client.get("/pages/folio/annotation")
    assert get.status_code == 200
    segments = get.json()["segments"]
    assert len(segments) == 2
    assert segments[0]["id"] == "seg-1" and segments[0]["kind"] == "polygon"
    assert segments[1]["id"] == "seg-2" and segments[1]["kind"] == "rectangle"
    assert len(segments[0]["points"]) == 4 and len(segments[1]["points"]) == 4


def test_annotation_empty_page(client, data_root):
    _seed_page(data_root)

    get = client.get("/pages/folio/annotation")
    assert get.status_code == 200
    assert get.json()["segments"] == []


def test_segment_defaults_source_manual_and_null_kraken_ceiling():
    segment = Segment(
        id="seg-1",
        number=1,
        kind="polygon",
        points=[[0, 0], [10, 0], [10, 5], [0, 5]],
    )
    assert segment.source == "manual"
    assert segment.kraken_ceiling is None


def test_page_annotation_accepts_legacy_segments_without_source_fields():
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
    assert annotation.segments[0].source == "manual"
    assert annotation.segments[0].kraken_ceiling is None


def test_annotation_corrupt_json_returns_422(client, data_root):
    _seed_page(data_root)
    ann_path = data_root / "annotations" / "pages" / "folio.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text("{not json", encoding="utf-8")

    response = client.get("/pages/folio/annotation")
    assert response.status_code == 422
