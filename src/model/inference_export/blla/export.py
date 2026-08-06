"""Development-only export of the inference-owned BLLA graph.

Restored by ADR 0006. ``_ExportGroupNorm`` below is the reason this file is
worth keeping: see that ADR for the accumulator bug it exists to avoid, and for
what shipping a graph exported *without* it cost.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from src.model.inference_export.blla.checkpoint import load_blla_model
from src.model.inference_export.blla.model import BLLATorchModel, _GroupNorm


class _ExportGroupNorm(nn.Module):
    """Group normalisation with a two-stage moment reduction, for ONNX export.

    ``nn.GroupNorm`` lowers to ``Reshape([0, 32, -1]) -> InstanceNormalization``,
    which flattens each group into a single axis of ``C/G * H * W`` elements.
    onnxruntime's CPU kernel accumulates that group's mean and variance in one
    float32 accumulator, so a 1800x2471 manuscript page puts 2,224,800 values
    through a single serial reduction. On real, spatially correlated post-ReLU
    activations the rounding bias accumulates instead of cancelling: the
    recovered per-group sigma drifts by ~1.2e-03 relative, the logits move by up
    to 0.89, and the handful of pixels that cross the 0.5 sigmoid boundary
    restructure short line polygons entirely (IoU 0.50 against the oracle).

    ``torch.nn.functional.group_norm`` uses a blocked/Welford accumulation and
    stays within float32 rounding noise, which is why only the ONNX runtime is
    affected. Reducing over the width axis first and then over the remaining
    group axis computes the identical arithmetic mean -- every block has exactly
    ``width`` elements -- but no accumulator ever sees more than a few thousand
    terms. Both stages lower to ``ReduceMean``, whose error onnxruntime then
    holds flat in the reduction length.

    This module is used *only* while tracing. ``BLLATorchModel`` keeps calling
    ``nn.GroupNorm`` at runtime, because the staged reduction perturbs the native
    float32 logits by up to 1.8e-03 -- harmless numerically, but enough to break
    the native decoder's bit-exact agreement with the Kraken oracle.

    Two stages is enough, and a third would buy nothing. The residual ONNX/Torch
    disagreement does grow with the scaled width, but it is not this reduction
    that drifts: exporting ``Gn_13`` -- the layer whose channels-per-group is 1,
    so the width is what dominates its reduction -- with the moments accumulated
    in float64 instead of float32 reproduces the *same* disagreement to six
    figures (4.63e-05 relative at a 3600-wide feature map, either way). An
    exactly-accumulated graph is no closer to Torch than this one, because the
    gap is Torch's own float32 ``group_norm`` at that reduction size. The lever
    that does work is the scaled width itself, and it lives in
    ``inference/architectures/blla/blla_preprocessing.py``.
    """

    def __init__(self, layer: nn.GroupNorm) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.float()
        batch, channels, height, width = values.shape
        grouped = values.reshape(batch, self.layer.num_groups, -1, width)
        mean = grouped.mean(dim=3).mean(dim=2)[:, :, None, None]
        centred = grouped - mean
        variance = centred.pow(2).mean(dim=3).mean(dim=2)[:, :, None, None]
        normalized = centred * torch.rsqrt(variance + self.layer.eps)
        normalized = normalized.reshape(batch, channels, height, width)
        scaled = normalized * self.layer.weight.view(1, -1, 1, 1)
        return (scaled + self.layer.bias.view(1, -1, 1, 1)).to(dtype=inputs.dtype)


def _with_export_group_norm(model: nn.Module) -> nn.Module:
    """Copy ``model`` with every ``_GroupNorm`` swapped for the stable form.

    The replacement keeps the original ``nn.GroupNorm`` as its ``layer`` child,
    so parameter names, the state dict, and ``blla.safetensors`` are untouched.
    """

    exportable = copy.deepcopy(model)
    for parent in exportable.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, _GroupNorm):
                setattr(parent, name, _ExportGroupNorm(child.layer))
    return exportable.eval()


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

    model = load_blla_model(str(source))
    if not isinstance(model, BLLATorchModel):
        raise TypeError("unexpected BLLA model loaded for export")

    destination.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(
        (1, model.input_channels, model.input_height, example_width),
        dtype=torch.float32,
    )
    torch.onnx.export(
        _with_export_group_norm(model),
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
