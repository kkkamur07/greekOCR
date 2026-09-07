"""Thin adapter around the optional `calamari_ocr` predict entry point.

Training-stack, not the inference runtime. See the package docstring: what runs
a page on a researcher's machine is `nomikos_inference.architectures.calamari`,
which opens an ONNX session and imports none of this.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


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
            *[
                argument
                for checkpoint in self._checkpoints
                for argument in ("--checkpoint", checkpoint)
            ],
            "--data.images",
            *[str(Path(image).expanduser().resolve()) for image in images],
            "--output_dir",
            str(Path(output_dir).expanduser().resolve()),
        ]
        # argv is a list, never a shell string, and every element is either this
        # interpreter, a literal flag, or a path this class already resolved.
        # There is no interpolation for an attacker to reach.
        subprocess.run(command, check=True)  # noqa: S603
