"""Config parsing for the minimal PyTorch Calamari graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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


def load_calamari_config(checkpoint_path: Path) -> CalamariTorchConfig:
    """Load the Calamari model graph config from a checkpoint JSON sidecar."""
    json_path = (
        checkpoint_path
        if checkpoint_path.suffix == ".json"
        else checkpoint_path.parent / f"{checkpoint_path.name}.json"
    )

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    model = payload["scenario"]["model"]

    return CalamariTorchConfig(
        layers=tuple(_parse_layer_config(layer) for layer in model["layers"]),
        classes=int(model["classes"]),
        temperature=float(model.get("temperature", -1.0)),
    )


def maxpool_strides(config: CalamariTorchLayerConfig) -> tuple[int, int]:
    pool_size = require_tuple(config.pool_size, config.name, "pool_size")
    raw_strides = require_tuple(config.strides, config.name, "strides")
    return tuple(
        pool if stride < 0 else stride
        for stride, pool in zip(raw_strides, pool_size, strict=True)
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


def _parse_layer_config(layer: dict[str, Any]) -> CalamariTorchLayerConfig:
    cls = str(layer["__cls__"])
    if cls.endswith(":Conv2DLayerParams"):
        return CalamariTorchLayerConfig(
            kind="conv2d",
            name=str(layer["name"]),
            filters=int(layer["filters"]),
            kernel_size=_parse_int_vec(layer["kernel_size"]),
            strides=_parse_int_vec(layer["strides"]),
            padding=str(layer["padding"]),
            activation=layer.get("activation"),
        )
    if cls.endswith(":MaxPool2DLayerParams"):
        return CalamariTorchLayerConfig(
            kind="maxpool2d",
            name=str(layer["name"]),
            pool_size=_parse_int_vec(layer["pool_size"]),
            strides=_parse_int_vec(layer["strides"]),
            padding=str(layer["padding"]),
        )
    if cls.endswith(":BiLSTMLayerParams"):
        return CalamariTorchLayerConfig(
            kind="bilstm",
            name=str(layer["name"]),
            hidden_nodes=int(layer["hidden_nodes"]),
            merge_mode=str(layer["merge_mode"]),
        )
    if cls.endswith(":DropoutLayerParams"):
        return CalamariTorchLayerConfig(
            kind="dropout",
            name=str(layer["name"]),
            rate=float(layer["rate"]),
        )
    raise ValueError(f"unsupported Calamari layer class: {cls}")


def _parse_int_vec(value: dict[str, Any]) -> tuple[int, int]:
    return int(value["x"]), int(value["y"])
