"""Calamari-compatible CNN–BiLSTM recognizer implemented in PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import CalamariTorchConfig, require_int
from .layers import LazyBiLSTM, SameConv2d, SameMaxPool2d, cnn_to_sequence


class CalamariTorchModel(nn.Module):
    """Recognize NHWC grayscale line images with CTC logits."""

    def __init__(self, config: CalamariTorchConfig) -> None:
        super().__init__()
        if config.classes < 2:
            raise ValueError("Calamari requires at least blank and one character class.")
        self.config = config
        self.layers = nn.ModuleList()
        channels = 1
        for layer_config in config.layers:
            if layer_config.kind == "conv2d":
                layer = SameConv2d(input_channels=channels, config=layer_config)
                channels = require_int(layer_config.filters, layer_config.name, "filters")
            elif layer_config.kind == "maxpool2d":
                layer = SameMaxPool2d(layer_config)
            elif layer_config.kind == "bilstm":
                layer = LazyBiLSTM(layer_config)
            elif layer_config.kind == "dropout":
                layer = nn.Dropout(p=float(layer_config.rate or 0.0))
            else:
                raise ValueError(f"Unsupported Calamari layer kind: {layer_config.kind}")
            self.layers.append(layer)
        self.logits = nn.LazyLinear(config.classes)

    def forward(
        self, image: Tensor, image_lengths: Tensor | None = None
    ) -> dict[str, Tensor]:
        if image.ndim != 4:
            raise ValueError("Calamari images must have shape (batch, time, height, channels).")
        value = image.to(dtype=torch.float32) / 255.0
        if image_lengths is None:
            image_lengths = torch.full(
                (value.shape[0],), value.shape[1], dtype=torch.long, device=value.device
            )
        output_lengths = self.config.downscaled_sequence_lengths(image_lengths)

        value = value.permute(0, 3, 1, 2)
        for layer in self.layers:
            if isinstance(layer, LazyBiLSTM):
                if value.ndim == 4:
                    value = cnn_to_sequence(value)
                value = layer(value, output_lengths)
            else:
                value = layer(value)
        if value.ndim == 4:
            value = cnn_to_sequence(value)

        blank_last_logits = self.logits(value)
        logits = torch.roll(blank_last_logits, shifts=1, dims=-1)
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        return {
            "blank_last_logits": blank_last_logits,
            "logits": logits,
            "out_len": output_lengths,
        }
