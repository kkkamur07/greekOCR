"""Import page images (and optional transcriptions) into the data layout."""

import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

from annote.services.page_catalogue import IMAGE_EXTENSIONS, build_page_summary

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tif",
}


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip()
    cleaned = re.sub(r"[^\w\-.]+", "_", stem).strip("._")
    return cleaned or "page"


def _decode_utf8_text(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Transcription file must be UTF-8 text: {filename}",
        ) from exc


def _extension_for_upload(upload: UploadFile) -> str:
    if upload.filename:
        ext = Path(upload.filename).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return ext
    if upload.content_type in ALLOWED_IMAGE_CONTENT_TYPES:
        return ALLOWED_IMAGE_CONTENT_TYPES[upload.content_type]
    raise HTTPException(status_code=400, detail="Unsupported image type. Use JPEG, PNG, WebP, or TIFF.")


async def import_page(
    data_root: Path,
    image: UploadFile,
    transcription: UploadFile | None = None,
) -> str:
    """Save uploaded files; return the page stem."""
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image file is required.")

    stem = safe_stem(image.filename)
    pages_dir = data_root / "manuscripts" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    ext = _extension_for_upload(image)
    dest = pages_dir / f"{stem}{ext}"
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Image file is empty.")
    dest.write_bytes(data)

    if transcription and transcription.filename:
        tx_dir = data_root / "transcriptions" / "pages"
        tx_dir.mkdir(parents=True, exist_ok=True)
        tx_bytes = await transcription.read()
        text = _decode_utf8_text(tx_bytes, transcription.filename)
        (tx_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

    return stem


def import_summary(data_root: Path, stem: str):
    return build_page_summary(data_root, stem)
