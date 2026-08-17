"""Inference adapter for canonical PyTorch Calamari checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy
import torch
from PIL import Image

from ..models.calamari.checkpoint import load_calamari_checkpoint
from ..models.calamari.codec import CharacterCodec


class CalamariPredictor:
    """Run one canonical PyTorch Calamari checkpoint on line image files."""

    def __init__(self, checkpoints: Sequence[str | Path]) -> None:
        if len(checkpoints) != 1:
            raise ValueError("PyTorch Calamari prediction currently supports exactly one checkpoint.")
        self._model, metadata = load_calamari_checkpoint(
            Path(checkpoints[0]).expanduser().resolve()
        )
        self._codec = CharacterCodec(metadata.charset)
        self._line_height = metadata.line_height

    def predict_files(self, images: Sequence[str | Path], *, output_dir: str | Path) -> None:
        """Write one prediction per input image to ``output_dir``."""
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            path = Path(image_path).expanduser().resolve()
            image = _load_image(path, self._line_height)
            with torch.no_grad():
                output = self._model(image, torch.tensor([image.shape[1]]))
            text = self._codec.decode_logits(output["logits"], output["out_len"])[0]
            (destination / f"{path.stem}.txt").write_text(text, encoding="utf-8")


def _load_image(path: Path, line_height: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        width = max(1, round(image.width * line_height / image.height))
        image = image.resize((width, line_height), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(numpy.asarray(image).T.copy()).unsqueeze(-1)
    return pixels.unsqueeze(0)
