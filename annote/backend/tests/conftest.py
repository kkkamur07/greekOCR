"""Shared pytest fixtures for annote backend."""

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from annote.app import create_app
from annote.settings import get_settings


def minimal_jpeg_bytes(width: int = 100, height: int = 50, color: str = "white") -> bytes:
    """Create a small JPEG in memory for integration tests."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return buf.getvalue()


def minimal_png_bytes(width: int = 100, height: int = 50, color: str = "white") -> bytes:
    """Create a small PNG in memory for integration tests."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Isolated data directory with expected layout."""
    for sub in (
        "manuscripts/pages",
        "manuscripts/export",
        "transcriptions/pages",
        "annotations/pages",
        "manuscripts/share",
    ):
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def unicode_font(monkeypatch: pytest.MonkeyPatch):
    """Ensure PDF tests can render Greek without skipping when a system font exists."""
    from annote.services.fonts import resolve_unicode_font
    import annote.services.transcription_pdf as transcription_pdf

    try:
        font_path = resolve_unicode_font()
    except RuntimeError as e:
        pytest.skip(str(e))

    transcription_pdf._font_registered = False
    monkeypatch.setattr(transcription_pdf, "resolve_unicode_font", lambda: font_path)
    monkeypatch.setattr("annote.services.fonts.resolve_unicode_font", lambda: font_path)
    return font_path


@pytest.fixture
def client(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with annote data root pointed at a temp directory."""
    monkeypatch.setenv("ANNOTE_DATA_ROOT", str(data_root))
    get_settings.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
