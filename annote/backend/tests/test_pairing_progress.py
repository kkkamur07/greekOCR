"""Pairing progress on page catalogue."""

from pathlib import Path

from annote.services.segment_text import compute_pairing_progress
from annote.services.text_lines import split_text_lines
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


def _seed_page(data_root: Path, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes(200, 120))
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text(
        "line one\nline two\nline three\n",
        encoding="utf-8",
    )


def test_text_override_counts_as_paired_without_line_index():
    from annote.schemas.annotation import Segment

    override_only = {
        **SEG2,
        "paired_text_line_index": None,
        "text_override": "typed directly",
    }
    progress = compute_pairing_progress([Segment(**override_only)], [])

    assert progress.paired_count == 1
    assert progress.unpaired_count == 0


def test_compute_pairing_progress_counts():
    from annote.schemas.annotation import Segment

    lines = split_text_lines("line one\nline two\nline three\n")
    progress = compute_pairing_progress([Segment(**SEG1), Segment(**SEG2)], lines)

    assert progress.paired_count == 1
    assert progress.unpaired_count == 1
    assert progress.text_line_count == 3
    assert progress.unused_line_count == 2


def test_list_pages_includes_pairing_progress(client, data_root):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None},
    )

    response = client.get("/pages")
    page = next(p for p in response.json()["pages"] if p["stem"] == "folio")

    assert page["pairing"]["paired_count"] == 1
    assert page["pairing"]["unpaired_count"] == 1
    assert page["pairing"]["text_line_count"] == 3
    assert page["pairing"]["unused_line_count"] == 2
