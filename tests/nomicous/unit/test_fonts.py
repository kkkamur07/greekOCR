"""Bundled Unicode font resolution for Transcription PDF."""

from pathlib import Path

from backend.core.fonts import resolve_unicode_font


def test_resolve_unicode_font_prefers_bundled_noto() -> None:
    font = resolve_unicode_font()
    assert font.is_file()
    assert font.name == "NotoSans-Regular.ttf"
    assert "assets" in font.parts
    assert font.stat().st_size > 0
    assert Path(font).suffix == ".ttf"
