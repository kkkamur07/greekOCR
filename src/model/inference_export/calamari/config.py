"""Configuration primitives for the PyTorch Calamari CNN–BiLSTM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

LayerKind = Literal["conv2d", "maxpool2d", "bilstm", "dropout"]


@dataclass(frozen=True)
class CalamariTorchLayerConfig:
    kind: LayerKind
    name: str
    filters: int | None = None
    kernel_size: tuple[int, int] | None = None
    strides: tuple[int, int] | None = None
    padding: str | None = None
    activation: str | None = None
    pool_size: tuple[int, int] | None = None
    hidden_nodes: int | None = None
    merge_mode: str | None = None
    rate: float | None = None


@dataclass(frozen=True)
class CalamariTorchConfig:
    layers: tuple[CalamariTorchLayerConfig, ...]
    classes: int
    temperature: float = -1.0

    def downscaled_sequence_lengths(self, sequence_lengths: Tensor) -> Tensor:
        lengths = sequence_lengths.to(dtype=torch.long)
        for layer in self.layers:
            if layer.kind == "conv2d":
                stride = require_tuple(layer.strides, layer.name, "strides")[0]
                lengths = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
            elif layer.kind == "maxpool2d":
                stride = maxpool_strides(layer)[0]
                lengths = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
        return lengths


def default_model_config(
    *, classes: int, temperature: float = -1.0, lstm_layers: int = 2
) -> CalamariTorchConfig:
    """Return the established Calamari CNN–BiLSTM topology."""
    if lstm_layers not in {1, 2}:
        raise ValueError("Calamari supports one or two bidirectional LSTM layers.")
    recurrent_layers = (
        CalamariTorchLayerConfig("bilstm", "lstm_0", hidden_nodes=200, merge_mode="concat"),
        CalamariTorchLayerConfig("dropout", "dropout_0", rate=0.3),
    )
    if lstm_layers == 2:
        recurrent_layers += (
            CalamariTorchLayerConfig("bilstm", "lstm_1", hidden_nodes=200, merge_mode="concat"),
        )
    return CalamariTorchConfig(
        layers=(
            CalamariTorchLayerConfig("conv2d", "conv2d_0", 40, (3, 3), (1, 1), "same", "relu"),
            CalamariTorchLayerConfig(
                "maxpool2d", "maxpool2d_0", pool_size=(2, 2), strides=(-1, -1), padding="same"
            ),
            CalamariTorchLayerConfig("conv2d", "conv2d_1", 60, (3, 3), (1, 1), "same", "relu"),
            CalamariTorchLayerConfig(
                "maxpool2d", "maxpool2d_1", pool_size=(2, 2), strides=(-1, -1), padding="same"
            ),
            *recurrent_layers,
        ),
        classes=classes,
        temperature=temperature,
    )


def maxpool_strides(config: CalamariTorchLayerConfig) -> tuple[int, int]:
    pool_size = require_tuple(config.pool_size, config.name, "pool_size")
    raw_strides = require_tuple(config.strides, config.name, "strides")
    return tuple(
        pool if stride < 0 else stride for stride, pool in zip(raw_strides, pool_size, strict=True)
    )


def require_int(value: int | None, layer_name: str, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{layer_name}.{field_name} is required")
    return value


def require_tuple(
    value: tuple[int, int] | None, layer_name: str, field_name: str
) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"{layer_name}.{field_name} is required")
    return value
