"""Load a tensor-only Calamari checkpoint into the runtime Torch graph.

Loading a pickled checkpoint executes code, so this module never unpickles:
``torch.load`` runs with ``weights_only=True``, and the caller has already
verified the **artifact SHA-256** through ``architectures.artifact`` before the
path reaches here.

This file was the loader half of the retired ONNX exporter
(``src/model/inference_export/calamari/export.py``). Under ADR 0004 the Torch
graph *is* the runtime, so the loader moved into the inference package and the
exporter was archived. ADR 0006 reversed that: the graph runs as ``.onnx``, and
both this loader and the exporter beside it are export-time code again.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from src.model.inference_export.calamari.config import (
    CalamariTorchConfig,
    CalamariTorchLayerConfig,
)
from src.model.inference_export.calamari.model import CalamariTorchModel
from torch import Tensor


@dataclass(frozen=True)
class CalamariCheckpointMetadata:
    """Everything the decoder needs from a checkpoint that is not a weight."""

    classes: int
    line_height: int
    charset: tuple[str, ...]
    blank_index: int = 0
    temperature: float = -1.0


def load_calamari_checkpoint(
    checkpoint_path: Path,
) -> tuple[CalamariTorchModel, CalamariCheckpointMetadata]:
    """Load and materialize a tensor-only Calamari checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"unable to safely load Calamari checkpoint: {checkpoint_path}") from error

    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "calamari-pytorch-v1":
        raise ValueError("unsupported Calamari checkpoint format")
    classes = checkpoint.get("classes")
    line_height = checkpoint.get("line_height", 48)
    charset = checkpoint.get("charset")
    state_dict = checkpoint.get("state_dict")
    if (
        not isinstance(classes, int)
        or isinstance(classes, bool)
        or classes < 2
        or not isinstance(line_height, int)
        or isinstance(line_height, bool)
        or line_height < 1
        or not isinstance(charset, list)
        or len(charset) != classes
        or not all(isinstance(character, str) for character in charset)
        or not isinstance(state_dict, Mapping)
        or not state_dict
        or not all(
            isinstance(name, str) and isinstance(value, Tensor)
            for name, value in state_dict.items()
        )
    ):
        raise ValueError("invalid Calamari checkpoint metadata or state dictionary")

    temperature = checkpoint.get("temperature", -1.0)
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
    ):
        raise ValueError("invalid Calamari checkpoint temperature")
    blank_index = checkpoint.get("blank_index", 0)
    if not isinstance(blank_index, int) or isinstance(blank_index, bool) or blank_index != 0:
        raise ValueError("only blank-index zero is supported by the Calamari runtime")

    metadata = CalamariCheckpointMetadata(
        classes=classes,
        line_height=line_height,
        charset=tuple(charset),
        blank_index=blank_index,
        temperature=float(temperature),
    )
    model = CalamariTorchModel(_default_config(metadata))
    model.eval()
    # Materialize LazyBiLSTM and LazyLinear before loading the state dict.  The
    # time width is deliberately arbitrary; weights do not depend on it.
    dummy = torch.zeros((1, 8, line_height, 1), dtype=torch.float32)
    # ``inference_mode`` would create inference tensors for the Lazy* parameters,
    # which cannot later receive a state-dict copy on recent Torch versions.
    with torch.no_grad():
        model(dummy, image_lengths=torch.tensor([8]))
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Calamari checkpoint state dictionary is incompatible") from error
    # Second ``eval()``: materializing the lazy modules above ran a forward
    # pass, and dropout must be off for every inference call that follows.
    model.eval()
    return model, metadata


def _default_config(metadata: CalamariCheckpointMetadata) -> CalamariTorchConfig:
    return CalamariTorchConfig(
        layers=(
            CalamariTorchLayerConfig(
                kind="conv2d",
                name="conv2d_0",
                filters=40,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            CalamariTorchLayerConfig(
                kind="maxpool2d",
                name="maxpool2d_0",
                pool_size=(2, 2),
                strides=(-1, -1),
                padding="same",
            ),
            CalamariTorchLayerConfig(
                kind="conv2d",
                name="conv2d_1",
                filters=60,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            CalamariTorchLayerConfig(
                kind="maxpool2d",
                name="maxpool2d_1",
                pool_size=(2, 2),
                strides=(-1, -1),
                padding="same",
            ),
            CalamariTorchLayerConfig(
                kind="bilstm",
                name="lstm_0",
                hidden_nodes=200,
                merge_mode="concat",
            ),
            CalamariTorchLayerConfig(
                kind="dropout",
                name="dropout_0",
                rate=0.5,
            ),
        ),
        classes=metadata.classes,
        temperature=metadata.temperature,
    )


__all__ = [
    "CalamariCheckpointMetadata",
    "load_calamari_checkpoint",
]
