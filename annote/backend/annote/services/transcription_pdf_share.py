"""Frozen transcription PDF written at page lock."""

from pathlib import Path

from annote.settings import get_settings


def share_pdf_path(data_root: Path, stem: str) -> Path:
    settings = get_settings().transcription_pdf
    filename = settings.share_filename_pattern.format(stem=stem)
    return data_root / settings.share_dir / filename


def write_share_pdf_bytes(data_root: Path, stem: str, pdf_bytes: bytes) -> Path:
    path = share_pdf_path(data_root, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes)
    return path


def remove_share_pdf(data_root: Path, stem: str) -> None:
    path = share_pdf_path(data_root, stem)
    if path.is_file():
        path.unlink()


def read_share_pdf(data_root: Path, stem: str) -> bytes | None:
    path = share_pdf_path(data_root, stem)
    if not path.is_file():
        return None
    return path.read_bytes()
