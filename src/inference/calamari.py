"""Thin production adapter around the optional Calamari runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


class CalamariPredictor:
    """Run Calamari inference without exposing its training internals."""

    def __init__(self, checkpoints: Sequence[str | Path]) -> None:
        if not checkpoints:
            raise ValueError("At least one Calamari checkpoint is required.")
        self._checkpoints = [str(Path(path).expanduser().resolve()) for path in checkpoints]

    def predict_files(self, images: Sequence[str | Path], *, output_dir: str | Path) -> None:
        """Write one prediction per input image to ``output_dir``."""
        command = [
            sys.executable,
            "-m",
            "calamari_ocr.scripts.predict",
            *[argument for checkpoint in self._checkpoints for argument in ("--checkpoint", checkpoint)],
            "--data.images",
            *[str(Path(image).expanduser().resolve()) for image in images],
            "--output_dir",
            str(Path(output_dir).expanduser().resolve()),
        ]
        subprocess.run(command, check=True)
