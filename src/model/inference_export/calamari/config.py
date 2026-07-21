"""Configuration for the reference Calamari graph."""

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
    value: tuple[int, int] | None,
    layer_name: str,
    field_name: str,
) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"{layer_name}.{field_name} is required")
    return value
