"""Development-only export of the inference-owned BLLA graph."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from inference.architectures.blla.blla import _load_blla_model
from inference.architectures.blla.blla_model import BLLATorchModel


def export_blla_onnx(
    source: Path,
    destination: Path,
    *,
    example_width: int = 64,
    opset_version: int = 18,
) -> None:
    """Export a validated native checkpoint with dynamic input width.

    This function intentionally owns no runtime fallback: the native Torch
    graph remains the parity oracle, while the generated artifact is consumed
    by the separate ONNX Runtime adapter.
    """

    if not source.is_file():
        raise FileNotFoundError(f"BLLA checkpoint not found: {source}")
    if source.suffix != ".safetensors":
        raise ValueError("BLLA ONNX export requires a safetensors checkpoint")
    if example_width <= 0:
        raise ValueError("example_width must be positive")
    if opset_version < 17:
        raise ValueError("opset_version must be at least 17")

    model = _load_blla_model(str(source))
    if not isinstance(model, BLLATorchModel):
        raise TypeError("unexpected BLLA model loaded for export")

    destination.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(
        (1, model.input_channels, model.input_height, example_width),
        dtype=torch.float32,
    )
    torch.onnx.export(
        model,
        example,
        destination,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset_version,
        dynamic_axes={
            "input": {3: "width"},
            "logits": {3: "output_width"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    _embed_blla_metadata(
        destination,
        source=source,
        input_height=model.input_height,
        input_channels=model.input_channels,
        output_channels=model.output_channels,
        opset_version=opset_version,
    )


def _embed_blla_metadata(
    destination: Path,
    *,
    source: Path,
    input_height: int,
    input_channels: int,
    output_channels: int,
    opset_version: int,
) -> None:
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("onnx is required to finalize a BLLA export") from error

    model = onnx.load(str(destination))
    metadata = {
        "format": "blla-onnx-v1",
        "graph": "inference-owned-blla-torch-v1",
        "input_layout": "NCHW",
        "input_channels": str(input_channels),
        "input_height": str(input_height),
        "output_channels": str(output_channels),
        "opset_version": str(opset_version),
        "dynamic_axes": "input.width,logits.output_width",
        "preprocessing": "RGB PIL Lanczos, float32 [0,1], global max inversion",
        "source_format": "blla-pytorch-v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))


__all__ = ["export_blla_onnx"]
