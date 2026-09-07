"""PyTorch Calamari OCR training, fine-tuning, and evaluation components."""

from .checkpoint import (
    CalamariCheckpointMetadata,
    load_calamari_checkpoint,
    save_calamari_checkpoint,
)
from .codec import CharacterCodec
from .config import CalamariTorchConfig, CalamariTorchLayerConfig, default_model_config
from .export import export_calamari_onnx
from .model import CalamariTorchModel

__all__ = [
    "CalamariCheckpointMetadata",
    "CalamariTorchConfig",
    "CalamariTorchLayerConfig",
    "CalamariTorchModel",
    "CharacterCodec",
    "default_model_config",
    "export_calamari_onnx",
    "load_calamari_checkpoint",
    "save_calamari_checkpoint",
]
"""Calamari engine boundary.

The supported ``calamari-ocr`` distribution supplies its runtime; training and
inference adapters live in :mod:`src.train` and :mod:`src.inference` so the
project does not vendor a second copy.
"""
