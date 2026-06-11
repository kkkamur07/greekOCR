"""High-quality image load/save helpers for export and display."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Browsers handle TIFF poorly; convert these to JPEG only for GET /pages/{stem}/image.
DISPLAY_JPEG_SOURCE_EXTENSIONS = {".tif", ".tiff"}


def needs_display_jpeg(image_path: Path) -> bool:
    """True when the page image should be transcoded to JPEG for browser display."""
    return image_path.suffix.lower() in DISPLAY_JPEG_SOURCE_EXTENSIONS


def encode_page_display_jpeg(image_path: Path) -> bytes:
    """Return high-quality JPEG bytes for editor display.

    The source file on disk (e.g. lossless TIFF) is never modified; export and OCR
    continue to read the original via load_page_rgb().
    """
    pil = Image.open(image_path)
    try:
        rgb = pil.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=95, subsampling=0)
        return buf.getvalue()
    finally:
        pil.close()


def load_page_rgb(image_path: Path) -> tuple[np.ndarray, Image.Image]:
    """Load a page image as an RGB numpy array, preserving metadata from the source."""
    pil = Image.open(image_path)
    try:
        rgb = pil.convert("RGB")
        return np.array(rgb), rgb
    finally:
        pil.close()


def _source_dpi(pil: Image.Image) -> tuple[int, int] | None:
    dpi = pil.info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) >= 2:
        x, y = dpi[0], dpi[1]
        if x > 0 and y > 0:
            return int(round(x)), int(round(y))
    return None


def save_line_image(image: np.ndarray, path: Path, *, source: Image.Image | None = None) -> None:
    """Save a processed line crop at maximum JPEG quality (4:4:4, no chroma subsampling)."""
    pil = Image.fromarray(image)
    dpi = _source_dpi(source) if source is not None else None
    save_kwargs: dict = {
        "format": "JPEG",
        "quality": 100,
        "subsampling": 0,
        "optimize": False,
    }
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    pil.save(path, **save_kwargs)
