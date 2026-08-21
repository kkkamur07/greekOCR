"""PyTorch layers used by the canonical Calamari CNN–BiLSTM model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import CalamariTorchLayerConfig, maxpool_strides, require_int, require_tuple


class SameConv2d(nn.Module):
    def __init__(self, *, input_channels: int, config: CalamariTorchLayerConfig) -> None:
        super().__init__()
        self.padding = config.padding or "valid"
        self.conv = nn.Conv2d(
            input_channels,
            require_int(config.filters, config.name, "filters"),
            kernel_size=require_tuple(config.kernel_size, config.name, "kernel_size"),
            stride=require_tuple(config.strides, config.name, "strides"),
            padding=0,
        )
        self.activation = _activation(config.activation)

    def forward(self, value: Tensor) -> Tensor:
        if self.padding == "same":
            value = _pad_same(value, self.conv.kernel_size, self.conv.stride)
        elif self.padding != "valid":
            raise ValueError(f"Unsupported Conv2D padding: {self.padding}")
        value = self.conv(value)
        return self.activation(value) if self.activation is not None else value


class SameMaxPool2d(nn.Module):
    def __init__(self, config: CalamariTorchLayerConfig) -> None:
        super().__init__()
        self.pool_size = require_tuple(config.pool_size, config.name, "pool_size")
        self.strides = maxpool_strides(config)
        self.padding = config.padding or "valid"

    def forward(self, value: Tensor) -> Tensor:
        if self.padding == "same":
            value = _pad_same(
                value, self.pool_size, self.strides, padding_value=float("-inf")
            )
        elif self.padding != "valid":
            raise ValueError(f"Unsupported MaxPool2D padding: {self.padding}")
        return functional.max_pool2d(value, kernel_size=self.pool_size, stride=self.strides)


class LazyBiLSTM(nn.Module):
    """Materialize after the CNN determines the feature width."""

    def __init__(self, config: CalamariTorchLayerConfig) -> None:
        super().__init__()
        self.hidden_nodes = require_int(config.hidden_nodes, config.name, "hidden_nodes")
        if (config.merge_mode or "concat") != "concat":
            raise ValueError(f"Unsupported BiLSTM merge mode: {config.merge_mode}")
        self.lstm: nn.LSTM | None = None

    def forward(self, value: Tensor, sequence_lengths: Tensor | None = None) -> Tensor:
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=value.shape[-1],
                hidden_size=self.hidden_nodes,
                batch_first=True,
                bidirectional=True,
            ).to(device=value.device, dtype=value.dtype)
        if sequence_lengths is None:
            return self.lstm(value)[0]
        if sequence_lengths.ndim != 1 or sequence_lengths.shape[0] != value.shape[0]:
            raise ValueError(
                "BiLSTM sequence lengths must contain one positive length per batch item."
            )
        if torch.any(sequence_lengths <= 0) or torch.any(sequence_lengths > value.shape[1]):
            raise ValueError(
                "BiLSTM sequence lengths must be positive and no greater than the time dimension."
            )
        packed = nn.utils.rnn.pack_padded_sequence(
            value,
            sequence_lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=value.shape[1],
        )
        return output


def cnn_to_sequence(value: Tensor) -> Tensor:
    batch, channels, time, height = value.shape
    return value.permute(0, 2, 3, 1).reshape(batch, time, height * channels)


def _pad_same(
    inputs: Tensor,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    *,
    padding_value: float = 0.0,
) -> Tensor:
    pad_time = _same_padding_amount(inputs.shape[-2], kernel_size[0], strides[0])
    pad_height = _same_padding_amount(inputs.shape[-1], kernel_size[1], strides[1])
    return functional.pad(
        inputs,
        (pad_height // 2, pad_height - pad_height // 2, pad_time // 2, pad_time - pad_time // 2),
        value=padding_value,
    )


def _same_padding_amount(size: int, kernel: int, stride: int) -> int:
    return max((math.ceil(size / stride) - 1) * stride + kernel - size, 0)


def _activation(name: str | None) -> nn.Module | None:
    if name is None:
        return None
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.1),
    }
    try:
        return activations[name]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error
