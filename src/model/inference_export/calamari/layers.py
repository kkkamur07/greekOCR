"""PyTorch layers matching the Calamari TensorFlow graph.

This module deliberately lives outside ``inference``.  It is the conversion
oracle, not a production runtime dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.inference_export.calamari.config import (
    CalamariTorchLayerConfig,
    maxpool_strides,
    require_int,
    require_tuple,
)


class SameConv2d(nn.Module):
    def __init__(self, *, input_channels: int, config: CalamariTorchLayerConfig) -> None:
        super().__init__()
        filters = require_int(config.filters, config.name, "filters")
        kernel_size = require_tuple(config.kernel_size, config.name, "kernel_size")
        strides = require_tuple(config.strides, config.name, "strides")
        self.padding = config.padding or "valid"
        self.conv = nn.Conv2d(
            input_channels,
            filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=0,
        )
        self.activation = activation(config.activation)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == "same":
            x = pad_same(x, self.conv.kernel_size, self.conv.stride)
        elif self.padding != "valid":
            raise ValueError(f"unsupported Conv2D padding: {self.padding}")
        x = self.conv(x)
        return self.activation(x) if self.activation is not None else x


class SameMaxPool2d(nn.Module):
    def __init__(self, config: CalamariTorchLayerConfig) -> None:
        super().__init__()
        self.pool_size = require_tuple(config.pool_size, config.name, "pool_size")
        self.strides = maxpool_strides(config)
        self.padding = config.padding or "valid"

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == "same":
            x = pad_same(x, self.pool_size, self.strides, value=float("-inf"))
        elif self.padding != "valid":
            raise ValueError(f"unsupported MaxPool2D padding: {self.padding}")
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.strides)


class LazyBiLSTM(nn.Module):
    def __init__(self, config: CalamariTorchLayerConfig) -> None:
        super().__init__()
        hidden_nodes = require_int(config.hidden_nodes, config.name, "hidden_nodes")
        if (config.merge_mode or "concat") != "concat":
            raise ValueError(f"unsupported BiLSTM merge_mode: {config.merge_mode}")
        self.hidden_nodes = hidden_nodes
        self.lstm: nn.LSTM | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=x.shape[-1],
                hidden_size=self.hidden_nodes,
                batch_first=True,
                bidirectional=True,
            ).to(device=x.device, dtype=x.dtype)
        return self.lstm(x)[0]


def cnn_to_sequence(x: Tensor) -> Tensor:
    batch, channels, time, height = x.shape
    return x.permute(0, 2, 3, 1).reshape(batch, time, height * channels)


def pad_same(
    x: Tensor,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    *,
    value: float = 0.0,
) -> Tensor:
    # ``x.shape`` becomes a Python constant in the legacy ONNX tracer.  Read
    # the shape tensor during export so temporal padding remains dynamic.
    if torch.onnx.is_in_onnx_export():
        shape = torch._shape_as_tensor(x)
        time_size: int | Tensor = shape[-2]
        height_size: int | Tensor = shape[-1]
    else:
        time_size = x.shape[-2]
        height_size = x.shape[-1]
    pad_time = _same_padding_amount(time_size, kernel_size[0], strides[0])
    pad_height = _same_padding_amount(height_size, kernel_size[1], strides[1])
    return F.pad(
        x,
        (
            pad_height // 2,
            pad_height - pad_height // 2,
            pad_time // 2,
            pad_time - pad_time // 2,
        ),
        value=value,
    )


def activation(name: str | None) -> nn.Module | None:
    if name is None:
        return None
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"unsupported activation: {name}")


def _same_padding_amount(size: int | Tensor, kernel: int, stride: int) -> int | Tensor:
    if isinstance(size, Tensor):
        output_size = torch.div(size + stride - 1, stride, rounding_mode="floor")
        return torch.clamp((output_size - 1) * stride + kernel - size, min=0)
    output_size = math.ceil(size / stride)
    return max((output_size - 1) * stride + kernel - size, 0)
