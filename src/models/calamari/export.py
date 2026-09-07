"""Export a PyTorch Calamari checkpoint as a dynamic-width ONNX model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .checkpoint import CalamariCheckpointMetadata, load_calamari_checkpoint
from .model import CalamariTorchModel


def export_calamari_onnx(
    checkpoint_path: Path, destination: Path, *, opset_version: int = 17
) -> CalamariCheckpointMetadata:
    """Export checkpoint weights and metadata for ONNX consumers."""
    model, metadata = load_calamari_checkpoint(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _OnnxWrapper(model).eval(),
        (
            torch.zeros((1, 8, metadata.line_height, 1), dtype=torch.float32),
            torch.tensor([8], dtype=torch.long),
        ),
        destination,
        input_names=["image", "image_lengths"],
        output_names=["logits", "out_len"],
        dynamic_axes={"image": {1: "time"}, "logits": {1: "time"}},
        opset_version=opset_version,
        dynamo=False,
    )
    import onnx

    onnx_model = onnx.load(destination)
    del onnx_model.metadata_props[:]
    for key, value in _metadata(metadata).items():
        property_value = onnx_model.metadata_props.add()
        property_value.key = key
        property_value.value = value
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, destination)
    return metadata


class _OnnxWrapper(nn.Module):
    def __init__(self, model: CalamariTorchModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor, image_lengths: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.model(image, image_lengths)
        return outputs["logits"], outputs["out_len"]


def _metadata(metadata: CalamariCheckpointMetadata) -> dict[str, str]:
    return {
        "format": "calamari-onnx-v1",
        "architecture": "calamari",
        "input_layout": "NHWC",
        "classes": str(metadata.classes),
        "line_height": str(metadata.line_height),
        "charset": json.dumps(list(metadata.charset), ensure_ascii=False),
        "blank_index": "0",
    }
