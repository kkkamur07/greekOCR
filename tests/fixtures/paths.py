"""Canonical paths and bytes for shared test fixtures."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = Path(__file__).resolve().parent
MANUSCRIPTS_DIR = FIXTURES_ROOT / "manuscripts"

# Minimal valid 1×1 PNG (same bytes as tests/nomicous/integration/test_documents.py)
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
    b"\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe"
    b"\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Script-scoped manuscript fixtures (script folder matches registry model script).
SEGMENT_PAGE = MANUSCRIPTS_DIR / "greek" / "segment_page.jpeg"
TRANSCRIBE_LINE = MANUSCRIPTS_DIR / "syriac" / "transcribe_line.jpg"


def segment_page_bytes() -> bytes:
    return SEGMENT_PAGE.read_bytes()


def transcribe_line_bytes() -> bytes:
    return TRANSCRIBE_LINE.read_bytes()
