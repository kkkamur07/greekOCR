"""Transcription PDF generation."""

import io
from pathlib import Path

from pypdf import PdfReader

from annote.services.transcription_pdf import generate_transcription_pdf
from annote.schemas.annotation import PageAnnotation
from annote.services.annotation_store import save_annotation
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
UNUSED_LINE = "unused transcription line"


def _seed_page(data_root: Path, stem: str = "folio"):
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes(200, 120))
    (data_root / "transcriptions" / "pages" / f"{stem}.txt").write_text(
        f"αἱρετικῶν\nκαὶ φιλοσόφων\n{UNUSED_LINE}\n",
        encoding="utf-8",
    )


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return reader.pages[0].extract_text() or ""


def test_transcription_pdf_endpoint_returns_pdf(client, data_root, unicode_font):
    _seed_page(data_root)
    client.put(
        "/pages/folio/annotation",
        json={"segments": [SEG1, SEG2], "export_metadata": None},
    )

    response = client.get("/pages/folio/transcription.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert "inline" in response.headers.get("content-disposition", "")


def test_transcription_pdf_includes_paired_text_only(data_root, unicode_font):
    _seed_page(data_root)
    from annote.schemas.annotation import Segment

    save_annotation(
        data_root,
        "folio",
        PageAnnotation(segments=[Segment(**SEG1), Segment(**SEG2), Segment(**UNPAIRED)]),
    )

    pdf_text = _pdf_text(generate_transcription_pdf(data_root, "folio"))

    assert "αἱρετικῶν" in pdf_text
    assert "καὶ φιλοσόφων" in pdf_text
    assert UNUSED_LINE not in pdf_text


def test_generate_transcription_pdf_service(data_root, unicode_font):
    _seed_page(data_root)
    from annote.schemas.annotation import Segment

    save_annotation(
        data_root,
        "folio",
        PageAnnotation(segments=[Segment(**SEG1), Segment(**SEG2)]),
    )

    pdf_bytes = generate_transcription_pdf(data_root, "folio")

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_transcription_pdf_blank_when_no_paired_segments(data_root, unicode_font):
    _seed_page(data_root)
    from annote.schemas.annotation import Segment

    save_annotation(
        data_root,
        "folio",
        PageAnnotation(segments=[Segment(**UNPAIRED)]),
    )

    pdf_bytes = generate_transcription_pdf(data_root, "folio")
    reader = PdfReader(io.BytesIO(pdf_bytes))

    assert len(reader.pages) == 1
    assert (reader.pages[0].extract_text() or "").strip() == ""
    assert reader.pages[0].mediabox.width == 200
    assert reader.pages[0].mediabox.height == 120


def test_transcription_pdf_matches_page_dimensions(data_root, unicode_font):
    _seed_page(data_root)
    from annote.schemas.annotation import Segment

    save_annotation(data_root, "folio", PageAnnotation(segments=[Segment(**SEG1)]))

    reader = PdfReader(io.BytesIO(generate_transcription_pdf(data_root, "folio")))

    assert reader.pages[0].mediabox.width == 200
    assert reader.pages[0].mediabox.height == 120


def _pdf_embedded_images(reader: PdfReader) -> list[object]:
    """Return image XObjects on the first page, if any."""
    page = reader.pages[0]
    resources = page.get("/Resources")
    if not resources:
        return []
    xobjects = resources.get("/XObject")
    if not xobjects:
        return []
    images: list[object] = []
    for _name, ref in xobjects.items():
        obj = ref.get_object()
        if obj.get("/Subtype") == "/Image":
            images.append(obj)
    return images


def test_transcription_pdf_has_no_facsimile_image(data_root, unicode_font):
    _seed_page(data_root)
    from annote.schemas.annotation import Segment

    save_annotation(
        data_root,
        "folio",
        PageAnnotation(segments=[Segment(**SEG1), Segment(**SEG2)]),
    )

    reader = PdfReader(io.BytesIO(generate_transcription_pdf(data_root, "folio")))

    assert _pdf_embedded_images(reader) == []
