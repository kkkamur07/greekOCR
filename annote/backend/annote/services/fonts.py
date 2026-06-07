"""Resolve a Unicode-capable TrueType font for PDF and image rendering."""

from pathlib import Path

_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/opt/homebrew/share/fonts/DejaVuSans.ttf"),
)


def resolve_unicode_font() -> Path:
    for candidate in _FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "No Unicode font found for PDF generation. "
        "Install DejaVu Sans (e.g. fonts-dejavu-core on Debian/Ubuntu)."
    )
