"""PAGE XML bundle naming and packaging, without a database.

The parts that touch Postgres and object storage are covered end to end in
``tests/nomikos/integration/test_transcription_pdf_artifact.py``; this file pins
the pure pieces: the shared file stem, the archive layout, and the download header.
"""

from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from backend.annotation.application.page_xml_export_service import (
    PageXmlBundle,
    export_file_stem,
)
from backend.core.api.content_disposition import attachment_disposition


@pytest.mark.parametrize(
    ("document_name", "page_number", "expected"),
    [
        ("My Codex", 3, "My_Codex_page_3"),
        (
            'slash/back\\colon:star*q?quote"lt<gt>pipe|',
            2,
            "slashbackcolonstarqquoteltgtpipe_page_2",
        ),
        ("Σιναϊτικός κώδικας", 12, "Σιναϊτικός_κώδικας_page_12"),
        ("", 1, "document_page_1"),
        ("trailing dots...", 4, "trailing_dots_page_4"),
    ],
)
def test_export_file_stem(document_name: str, page_number: int, expected: str) -> None:
    assert export_file_stem(document_name, page_number) == expected


def test_export_file_stem_caps_long_titles() -> None:
    stem = export_file_stem("x" * 500, 7)
    assert stem == "x" * 80 + "_page_7"


def test_bundle_zips_xml_and_image_side_by_side() -> None:
    bundle = PageXmlBundle(
        stem="Codex_page_2",
        xml=b"<?xml version='1.0'?><PcGts/>",
        image_filename="Codex_page_2.webp",
        image=b"RIFF....WEBP",
    )

    assert bundle.xml_filename == "Codex_page_2.xml"
    assert bundle.zip_filename == "Codex_page_2.zip"

    with ZipFile(BytesIO(bundle.to_zip())) as archive:
        assert archive.namelist() == ["Codex_page_2.xml", "Codex_page_2.webp"]
        assert archive.read("Codex_page_2.xml") == bundle.xml
        assert archive.read("Codex_page_2.webp") == bundle.image
        # The XML deflates, the already-compressed raster is stored as-is.
        assert archive.getinfo("Codex_page_2.xml").compress_type == ZIP_DEFLATED
        assert archive.getinfo("Codex_page_2.webp").compress_type == ZIP_STORED


def test_attachment_disposition_plain_ascii_has_no_encoded_form() -> None:
    assert attachment_disposition("Codex_page_1.zip") == 'attachment; filename="Codex_page_1.zip"'


def test_attachment_disposition_non_ascii_keeps_a_fallback_and_the_real_name() -> None:
    header = attachment_disposition("Σιναϊτικός_page_1.zip")
    assert header.startswith("attachment; filename=\"__page_1.zip\"; filename*=UTF-8''")
    assert "%CE%A3" in header  # Σ, percent-encoded UTF-8
    assert header.isascii()
