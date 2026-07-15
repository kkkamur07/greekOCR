"""Bundled Unicode font resolution for Transcription PDF."""

from pathlib import Path

import pytest

from backend.core import fonts
from backend.core.fonts import resolve_unicode_font


def test_resolve_unicode_font_prefers_bundled_noto() -> None:
    font = resolve_unicode_font()
    assert font.is_file()
    assert font.name == "NotoSans-Regular.ttf"
    assert "assets" in font.parts
    assert font.stat().st_size > 0
    assert Path(font).suffix == ".ttf"


def test_resolve_unicode_font_falls_back_to_os_font(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = tmp_path / "DejaVuSans.ttf"
    fallback.write_bytes(b"fake-font")
    monkeypatch.setattr(fonts, "_ASSETS_FONT", tmp_path / "missing.ttf")
    monkeypatch.setattr(fonts, "_OS_FONT_CANDIDATES", (fallback,))

    assert resolve_unicode_font() == fallback


def test_resolve_unicode_font_raises_plain_language_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fonts, "_ASSETS_FONT", tmp_path / "missing.ttf")
    monkeypatch.setattr(fonts, "_OS_FONT_CANDIDATES", ())

    with pytest.raises(RuntimeError, match="no Unicode font is available"):
        resolve_unicode_font()
