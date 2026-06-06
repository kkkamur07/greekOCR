"""Filesystem data layout for annote."""

from pathlib import Path

REQUIRED_SUBDIRS = (
    "manuscripts/pages",
    "manuscripts/export",
    "transcriptions/pages",
    "annotations/pages",
)


def list_required_subdirs() -> tuple[str, ...]:
    return REQUIRED_SUBDIRS


def export_dir(data_root: Path) -> Path:
    """Directory for exported line images and transcription files."""
    return data_root / "manuscripts" / "export"


def ensure_data_directories(data_root: Path) -> None:
    """Create PRD data subdirectories if missing."""
    data_root.mkdir(parents=True, exist_ok=True)
    for subdir in REQUIRED_SUBDIRS:
        (data_root / subdir).mkdir(parents=True, exist_ok=True)
