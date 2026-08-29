"""Download naming and the streamed zip sink, without a database.

The parts that touch Postgres and object storage are covered end to end in
``tests/nomikos/integration/test_document_exports.py``. This file pins the two pure
pieces: the slug a whole-document download is named by, and the write-only sink that
makes the PAGE XML archive stream instead of buffer.
"""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pytest

from backend.core.api.content_disposition import attachment_disposition
from backend.document.application.document_export_page_xml import _ZipSink
from backend.document.application.document_export_selection import document_download_name


@pytest.mark.parametrize(
    ("document_name", "expected"),
    [
        ("Chapter Four", "chapter-four-page-xml.zip"),
        # Whitespace runs and underscores are the same separator once slugged.
        ("Vat.  gr.   1360", "vat.-gr.-1360-page-xml.zip"),
        ("Chapter_Four", "chapter-four-page-xml.zip"),
        # A control character is stripped rather than treated as whitespace, which is
        # what ``export_file_stem`` does with the same character class. It is not a
        # separator anybody typed on purpose.
        ("Vat.\tgr.", "vat.gr-page-xml.zip"),
        # Path separators and the characters Windows refuses are dropped outright, and
        # the hyphen runs they leave behind collapse.
        ('a/b:c*d?e"f<g>h|i', "abcdefghi-page-xml.zip"),
        # A Greek title stays Greek; the header layer is what makes it transportable.
        ("Ιλιάς Α", "ιλιάς-α-page-xml.zip"),
        # A name with nothing usable left in it still has to produce a filename.
        ("///", "document-page-xml.zip"),
        ("", "document-page-xml.zip"),
        ("   ", "document-page-xml.zip"),
    ],
)
def test_download_name_slugs_the_document_title(document_name: str, expected: str) -> None:
    assert document_download_name(document_name, "page-xml.zip") == expected


def test_download_name_is_bounded_by_the_slug_cap() -> None:
    """A long title cannot push the download name past a path limit."""
    name = document_download_name("Ω" * 500, "page-xml.zip")
    assert name == "ω" * 80 + "-page-xml.zip"


def test_download_name_survives_the_content_disposition_header() -> None:
    """The header layer is Latin-1, so a Greek slug has to arrive as RFC 6266."""
    header = attachment_disposition(document_download_name("Ιλιάς Α", "transcription.txt"))
    assert header.startswith('attachment; filename="_-_-transcription.txt"')
    assert "filename*=UTF-8''" in header


# --- The streamed archive sink ---


def test_zip_sink_drains_each_entry_and_produces_a_readable_archive() -> None:
    """Bytes leave the sink as they are written, and the archive still opens.

    The drain-per-entry is what keeps peak memory at one page rather than the whole
    chapter, so it is checked directly: after each entry the sink hands over what it
    has and keeps nothing.
    """
    sink = _ZipSink()
    chunks: list[bytes] = []
    with ZipFile(sink, "w") as archive:
        archive.writestr("0001.xml", b"<PcGts/>")
        chunks.append(sink.drain())
        archive.writestr("0002.xml", b"<PcGts/>")
        chunks.append(sink.drain())
    chunks.append(sink.drain())

    assert all(chunk for chunk in chunks), "each entry should have flushed bytes"
    # Nothing is held back: the concatenated chunks are the whole archive.
    assert sink.drain() == b""
    with ZipFile(BytesIO(b"".join(chunks))) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["0001.xml", "0002.xml"]
        assert archive.read("0002.xml") == b"<PcGts/>"


def test_zip_sink_reports_a_position_but_refuses_to_seek() -> None:
    """``ZipFile`` needs ``tell`` to place offsets and needs ``seek`` to be absent.

    With both present it would rewrite local headers in place, which a response body
    already on the wire cannot do; with neither it wraps the sink and loses the count.
    """
    sink = _ZipSink()
    assert sink.tell() == 0
    assert sink.write(b"abcd") == 4
    assert sink.tell() == 4
    assert not hasattr(sink, "seek")
