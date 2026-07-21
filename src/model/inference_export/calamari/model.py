"""Forward pass for the reference Calamari graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.inference_export.calamari.config import CalamariTorchConfig, require_int
from src.model.inference_export.calamari.layers import (
    LazyBiLSTM,
    SameConv2d,
    SameMaxPool2d,
    cnn_to_sequence,
)


class CalamariTorchModel(nn.Module):
    """PyTorch equivalent of the common Calamari CNN-BiLSTM graph."""

    def __init__(self, config: CalamariTorchConfig) -> None:
        super().__init__()
        if config.classes <= 0:
            raise ValueError("CalamariTorchConfig.classes must be positive")
        self.config = config
        self.layers = nn.ModuleList()
        input_channels = 1

        for layer in config.layers:
            if layer.kind == "conv2d":
                module = SameConv2d(input_channels=input_channels, config=layer)
                input_channels = require_int(layer.filters, layer.name, "filters")
            elif layer.kind == "maxpool2d":
                module = SameMaxPool2d(layer)
            elif layer.kind == "bilstm":
                module = LazyBiLSTM(layer)
            elif layer.kind == "dropout":
                module = nn.Dropout(p=float(layer.rate or 0.0))
            else:
                raise ValueError(f"unsupported Calamari layer kind: {layer.kind}")
            self.layers.append(module)

        self.logits = nn.LazyLinear(config.classes)

    def forward(self, image: Tensor, image_lengths: Tensor | None = None) -> dict[str, Tensor]:
        """Run the Calamari graph on ``batch x time x height x channel`` input."""
        if image.ndim != 4:
            raise ValueError("image must be a 4D NHWC tensor")

        x = image.to(dtype=torch.float32) / 255.0
        if image_lengths is None:
            image_lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)

        x = x.permute(0, 3, 1, 2)
        for layer in self.layers:
            if isinstance(layer, LazyBiLSTM):
                x = cnn_to_sequence(x)
                x = layer(x)
            else:
                x = layer(x)

        if x.ndim == 4:
            x = cnn_to_sequence(x)

        blank_last_logits = self.logits(x)
        logits = torch.roll(blank_last_logits, shifts=1, dims=-1)
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        return {
            "blank_last_logits": blank_last_logits,
            "blank_last_softmax": torch.softmax(blank_last_logits, dim=-1),
            "out_len": self.config.downscaled_sequence_lengths(image_lengths),
            "logits": logits,
            "softmax": torch.softmax(logits, dim=-1),
        }
