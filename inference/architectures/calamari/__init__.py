"""Calamari OCR architecture adapters and PyTorch inference graph."""

from inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    run_calamari_transcribe,
    run_calamari_transcribe_many,
)
from inference.architectures.calamari.config import (
    CalamariTorchConfig,
    CalamariTorchLayerConfig,
)
from inference.architectures.calamari.model import CalamariTorchModel
from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_to_calamari_tensor,
)

__all__ = [
    "CalamariTorchConfig",
    "CalamariTorchLayerConfig",
    "CalamariTorchModel",
    "CalamariUnavailableError",
    "preprocess_line_image_to_calamari_tensor",
    "run_calamari_transcribe",
    "run_calamari_transcribe_many",
]
