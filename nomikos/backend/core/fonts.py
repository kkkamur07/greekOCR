"""Resolve Unicode-capable fonts for generated artifacts."""

from __future__ import annotations

from pathlib import Path

_ASSETS_FONT = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSans-Regular.ttf"

_OS_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/opt/homebrew/share/fonts/DejaVuSans.ttf"),
)

_FONT_MISSING_MESSAGE = (
    "Transcription PDF could not be generated because no Unicode font is available. "
    "The app expects its bundled Noto Sans font; contact support if this keeps happening."
)


def resolve_unicode_font() -> Path:
    """Return a Greek-capable Unicode TTF, preferring the bundled font."""
    if _ASSETS_FONT.is_file():
        return _ASSETS_FONT
    for candidate in _OS_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise RuntimeError(_FONT_MISSING_MESSAGE)
