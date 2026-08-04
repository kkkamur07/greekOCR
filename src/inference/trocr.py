"""Production inference adapter for trained Transformers TrOCR checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TrOCRPredictor:
    """Load one exported TrOCR checkpoint and transcribe line images."""

    def __init__(self, model: Any, processor: Any, device: Any) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | None = None,
    ) -> "TrOCRPredictor":
        """Load a checkpoint produced by ``greekocr-train-trocr``."""
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        processor = TrOCRProcessor.from_pretrained(checkpoint_path)
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(resolved_device).eval()
        return cls(model=model, processor=processor, device=resolved_device)

    def predict(self, image: Any, *, max_length: int = 50, num_beams: int = 1) -> str:
        """Transcribe one PIL image without importing any training components."""
        import torch

        pixel_values = self._processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        with torch.inference_mode():
            generated_ids = self._model.generate(
                pixel_values.to(self._device),
                max_length=max_length,
                num_beams=num_beams,
            )
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
