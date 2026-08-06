"""Export the reference Calamari graph as a self-contained ONNX artifact.

Restored by ADR 0006, which supersedes 0004.

Originally ``src/model/inference_export/calamari/export.py``. The checkpoint
*loader* that used to live beside this exporter was not retired with it - it
is how the Torch runtime opens ``best.pt`` today, and now lives at
``inference/architectures/calamari/checkpoint.py`` along with the graph
(``model.py``, ``layers.py``, ``config.py``) this file imported from
``src/model/inference_export/calamari/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from src.model.inference_export.calamari.checkpoint import (
    CalamariCheckpointMetadata,
    load_calamari_checkpoint,
)
from src.model.inference_export.calamari.model import CalamariTorchModel
from torch import Tensor, nn


def export_calamari_onnx(
    checkpoint_path: Path,
    destination: Path,
    *,
    opset_version: int = 17,
) -> CalamariCheckpointMetadata:
    """Export a converted Calamari checkpoint and return embedded metadata."""
    model, metadata = load_calamari_checkpoint(checkpoint_path)
    wrapper = _CalamariONNXWrapper(model).eval()
    dummy_image = torch.zeros(
        (1, 8, metadata.line_height, 1),
        dtype=torch.float32,
    )
    dummy_lengths = torch.tensor([8], dtype=torch.long)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_lengths),
            destination,
            input_names=["image", "image_lengths"],
            output_names=["logits", "out_len"],
            dynamic_axes={
                # The runtime submits one line at a time.  Keeping batch
                # static avoids the unsupported variable-batch LSTM state
                # warning while preserving arbitrary temporal widths.
                "image": {1: "time"},
                "logits": {1: "time"},
            },
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
        )
    except Exception as error:
        raise RuntimeError(f"unable to export Calamari ONNX artifact: {destination}") from error

    try:
        import onnx

        onnx_model = onnx.load(destination)
        del onnx_model.metadata_props[:]
        for key, value in _metadata_values(metadata).items():
            prop = onnx_model.metadata_props.add()
            prop.key = key
            prop.value = value
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, destination)
    except Exception as error:
        raise RuntimeError(f"unable to embed Calamari ONNX metadata: {destination}") from error
    return metadata


class _CalamariONNXWrapper(nn.Module):
    def __init__(self, model: CalamariTorchModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor, image_lengths: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.model(image, image_lengths=image_lengths)
        return outputs["logits"], outputs["out_len"]


def _metadata_values(metadata: CalamariCheckpointMetadata) -> dict[str, str]:
    return {
        "format": "calamari-onnx-v1",
        "architecture": "calamari",
        "input_layout": "NHWC",
        "input_name": "image",
        "output_names": json.dumps(["logits", "out_len"]),
        "classes": str(metadata.classes),
        "line_height": str(metadata.line_height),
        "charset": json.dumps(list(metadata.charset), ensure_ascii=False),
        "blank_index": str(metadata.blank_index),
        "temperature": repr(metadata.temperature),
        "preprocessing": "existing Calamari NumPy preprocessing; model input is uint8-valued float32",
    }
