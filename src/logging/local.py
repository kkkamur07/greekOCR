"""Local file logging kept separate from model artifacts."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_file_logging(*, log_file: Path, level: str = "INFO") -> logging.Logger:
    """Configure console and UTF-8 file logging for one command invocation."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("greekocr")
