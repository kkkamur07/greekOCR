"""Inference-owned PyTorch implementation of the shipped BLLA network."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _ConvActivation(nn.Module):
    """VGSL convolution with the BLLA activation and padding semantics."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        padding = tuple((size - 1) // 2 for size in kernel_size)
        self.co = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.relu(self.co(inputs))


class _GroupNorm(nn.Module):
    """Named wrapper matching the serialized BLLA module layout."""

    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        self.layer = nn.GroupNorm(groups, channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layer(inputs.float()).to(dtype=inputs.dtype)


class _LinearHead(nn.Module):
    """Linear 1×1 projection with the serialized ``co`` parameter name."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.co = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.co(inputs)


class _AxisBiLSTM(nn.Module):
    """Run a batch of independent bidirectional sequences on one image axis."""

    def __init__(self, input_size: int, *, transpose: bool) -> None:
        super().__init__()
        self.transpose = transpose
        self.layer = nn.LSTM(
            input_size=input_size,
            hidden_size=32,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # NCHW -> HNWC, then optionally HNWC -> WNHC.
        sequence = inputs.permute(2, 0, 3, 1)
        if self.transpose:
            sequence = sequence.transpose(0, 2)

        size = sequence.size()
        sequence = sequence.contiguous().view(-1, size[2], size[3])
        output, _ = self.layer(sequence)
        output = output.view(size[0], size[1], size[2], 64)
        if self.transpose:
            output = output.transpose(0, 2)

        # HNWC -> NCHW.
        return output.permute(1, 3, 0, 2)


class BLLATorchModel(nn.Module):
    """Fixed PyTorch graph for the native BLLA checkpoint.

    The model accepts ``NCHW`` tensors with three RGB channels. Its height is
    reduced from 1800 to 450 and its width is reduced by the same factor. The
    four output channels are, in order, start separator, end separator,
    baseline, and text-region logits.
    """

    input_height = 1800
    input_channels = 3
    output_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.C_0 = _ConvActivation(3, 64, (7, 7), (2, 2))
        self.Gn_1 = _GroupNorm(64)
        self.C_2 = _ConvActivation(64, 128, (3, 3), (2, 2))
        self.Gn_3 = _GroupNorm(128)
        self.C_4 = _ConvActivation(128, 128, (3, 3))
        self.Gn_5 = _GroupNorm(128)
        self.C_6 = _ConvActivation(128, 256, (3, 3))
        self.Gn_7 = _GroupNorm(256)
        self.C_8 = _ConvActivation(256, 256, (3, 3))
        self.Gn_9 = _GroupNorm(256)
        self.L_10 = _AxisBiLSTM(256, transpose=False)
        self.L_11 = _AxisBiLSTM(64, transpose=True)
        self.C_12 = _ConvActivation(64, 32, (1, 1))
        self.Gn_13 = _GroupNorm(32)
        self.L_14 = _AxisBiLSTM(32, transpose=True)
        self.L_15 = _AxisBiLSTM(64, transpose=False)
        self.l_16 = _LinearHead(64, 4)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("BLLA input must be a 4D NCHW tensor")
        if inputs.shape[1] != self.input_channels:
            raise ValueError("BLLA input must have three RGB channels")
        if inputs.shape[2] != self.input_height:
            raise ValueError("BLLA input height must be 1800 pixels")

        output = self.C_0(inputs)
        output = self.Gn_1(output)
        output = self.C_2(output)
        output = self.Gn_3(output)
        output = self.C_4(output)
        output = self.Gn_5(output)
        output = self.C_6(output)
        output = self.Gn_7(output)
        output = self.C_8(output)
        output = self.Gn_9(output)
        output = self.L_10(output)
        output = self.L_11(output)
        output = self.C_12(output)
        output = self.Gn_13(output)
        output = self.L_14(output)
        output = self.L_15(output)
        return self.l_16(output)
