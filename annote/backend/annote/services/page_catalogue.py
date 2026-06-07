"""Page catalogue — discover page images on disk."""

from pathlib import Path

from annote.schemas.pages import PageSummary
from annote.services.annotation_store import load_annotation
from annote.services.export_state import is_export_dirty
from annote.services.segment_text import compute_pairing_progress
from annote.services.text_lines import split_text_lines


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def image_media_type(path: Path) -> str:
    return IMAGE_MEDIA_TYPES[path.suffix.lower()]


def list_page_stems(pages_dir: Path) -> list[str]:
    if not pages_dir.is_dir():
        return []
    stems: list[str] = []
    for path in sorted(pages_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stems.append(path.stem)
    return stems


def resolve_page_image(pages_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = pages_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def has_transcription(transcriptions_dir: Path, stem: str) -> bool:
    return (transcriptions_dir / f"{stem}.txt").is_file()


def build_page_summary(data_root: Path, stem: str) -> PageSummary:
    annotation = load_annotation(data_root, stem)
    transcription_path = data_root / "transcriptions" / "pages" / f"{stem}.txt"
    raw_text = transcription_path.read_text(encoding="utf-8") if transcription_path.is_file() else ""
    text_lines = split_text_lines(raw_text) if raw_text.strip() else []
    return PageSummary(
        stem=stem,
        has_transcription=has_transcription(data_root / "transcriptions" / "pages", stem),
        segment_count=len(annotation.segments),
        export_dirty=is_export_dirty(data_root, stem, annotation),
        locked=annotation.locked,
        pairing=compute_pairing_progress(annotation.segments, text_lines),
    )


def list_pages(data_root: Path) -> list[PageSummary]:
    pages_dir = data_root / "manuscripts" / "pages"
    return [build_page_summary(data_root, stem) for stem in list_page_stems(pages_dir)]
