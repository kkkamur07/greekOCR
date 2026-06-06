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


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Isolated data directory with expected layout."""
    for sub in (
        "manuscripts/pages",
        "manuscripts/export",
        "transcriptions/pages",
        "annotations/pages",
    ):
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def client(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with annote data root pointed at a temp directory."""
    monkeypatch.setenv("ANNOTE_DATA_ROOT", str(data_root))
    get_settings.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
