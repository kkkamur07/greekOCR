"""Calamari OCR adapter with lazy reference-graph compatibility exports."""

from inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    run_calamari_transcribe,
    run_calamari_transcribe_many,
)
from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_to_calamari_tensor,
)


def __getattr__(name: str) -> object:
    if name in {"CalamariTorchConfig", "CalamariTorchLayerConfig"}:
        from src.model.inference_export.calamari.config import (
            CalamariTorchConfig,
            CalamariTorchLayerConfig,
        )

        return {
            "CalamariTorchConfig": CalamariTorchConfig,
            "CalamariTorchLayerConfig": CalamariTorchLayerConfig,
        }[name]
    if name == "CalamariTorchModel":
        from src.model.inference_export.calamari.model import CalamariTorchModel

        return CalamariTorchModel
    raise AttributeError(name)


__all__ = [
    "CalamariTorchConfig",
    "CalamariTorchLayerConfig",
    "CalamariTorchModel",
    "CalamariUnavailableError",
    "preprocess_line_image_to_calamari_tensor",
    "run_calamari_transcribe",
    "run_calamari_transcribe_many",
]
