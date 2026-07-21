"""Export the reference Calamari graph as a self-contained ONNX artifact."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from src.model.inference_export.calamari.config import (
    CalamariTorchConfig,
    CalamariTorchLayerConfig,
)
from src.model.inference_export.calamari.model import CalamariTorchModel


@dataclass(frozen=True)
class CalamariExportMetadata:
    classes: int
    line_height: int
    charset: tuple[str, ...]
    blank_index: int = 0
    temperature: float = -1.0


def load_calamari_checkpoint(
    checkpoint_path: Path,
) -> tuple[CalamariTorchModel, CalamariExportMetadata]:
    """Load and materialize a tensor-only Calamari checkpoint for export."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"unable to safely load Calamari checkpoint: {checkpoint_path}") from error

    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "calamari-pytorch-v1":
        raise ValueError("unsupported Calamari checkpoint format")
    classes = checkpoint.get("classes")
    line_height = checkpoint.get("line_height", 48)
    charset = checkpoint.get("charset")
    state_dict = checkpoint.get("state_dict")
    if (
        not isinstance(classes, int)
        or isinstance(classes, bool)
        or classes < 2
        or not isinstance(line_height, int)
        or isinstance(line_height, bool)
        or line_height < 1
        or not isinstance(charset, list)
        or len(charset) != classes
        or not all(isinstance(character, str) for character in charset)
        or not isinstance(state_dict, Mapping)
        or not state_dict
        or not all(
            isinstance(name, str) and isinstance(value, Tensor)
            for name, value in state_dict.items()
        )
    ):
        raise ValueError("invalid Calamari checkpoint metadata or state dictionary")

    temperature = checkpoint.get("temperature", -1.0)
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
    ):
        raise ValueError("invalid Calamari checkpoint temperature")
    blank_index = checkpoint.get("blank_index", 0)
    if not isinstance(blank_index, int) or isinstance(blank_index, bool) or blank_index != 0:
        raise ValueError("only blank-index zero is supported by the Calamari runtime")

    metadata = CalamariExportMetadata(
        classes=classes,
        line_height=line_height,
        charset=tuple(charset),
        blank_index=blank_index,
        temperature=float(temperature),
    )
    model = CalamariTorchModel(_default_config(metadata))
    model.eval()
    # Materialize LazyBiLSTM and LazyLinear before loading or exporting.  The
    # time width is deliberately arbitrary; weights do not depend on it.
    dummy = torch.zeros((1, 8, line_height, 1), dtype=torch.float32)
    # ``inference_mode`` would create inference tensors for Lazy* parameters,
    # which cannot later receive a state-dict copy on recent Torch versions.
    with torch.no_grad():
        model(dummy, image_lengths=torch.tensor([8]))
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Calamari checkpoint state dictionary is incompatible") from error
    model.eval()
    return model, metadata


def export_calamari_onnx(
    checkpoint_path: Path,
    destination: Path,
    *,
    opset_version: int = 17,
) -> CalamariExportMetadata:
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


def _default_config(metadata: CalamariExportMetadata) -> CalamariTorchConfig:
    return CalamariTorchConfig(
        layers=(
            CalamariTorchLayerConfig(
                kind="conv2d",
                name="conv2d_0",
                filters=40,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            CalamariTorchLayerConfig(
                kind="maxpool2d",
                name="maxpool2d_0",
                pool_size=(2, 2),
                strides=(-1, -1),
                padding="same",
            ),
            CalamariTorchLayerConfig(
                kind="conv2d",
                name="conv2d_1",
                filters=60,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            CalamariTorchLayerConfig(
                kind="maxpool2d",
                name="maxpool2d_1",
                pool_size=(2, 2),
                strides=(-1, -1),
                padding="same",
            ),
            CalamariTorchLayerConfig(
                kind="bilstm",
                name="lstm_0",
                hidden_nodes=200,
                merge_mode="concat",
            ),
            CalamariTorchLayerConfig(
                kind="dropout",
                name="dropout_0",
                rate=0.5,
            ),
        ),
        classes=metadata.classes,
        temperature=metadata.temperature,
    )


def _metadata_values(metadata: CalamariExportMetadata) -> dict[str, str]:
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
