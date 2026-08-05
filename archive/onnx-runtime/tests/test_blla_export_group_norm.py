"""Guards for the BLLA ONNX export's staged GroupNorm reduction.

RETIRED. See ``archive/onnx-runtime/README.md`` and ADR 0004. These lived in
``tests/inference/unit/test_blla.py`` next to the live BLLA tests, because the
export shim had to be proven equal to `nn.GroupNorm` *and* proven not to leak
into the runtime model. They moved here with the exporter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from inference.architectures.blla.blla_model import BLLATorchModel

REPO_ROOT = Path(__file__).resolve().parents[3]
BLLA_ONNX_ARTIFACT = (
    REPO_ROOT
    / "src"
    / "hf"
    / "staging"
    / "models"
    / "segmentation"
    / "blla"
    / "v1"
    / "stable"
    / "blla.onnx"
)


def test_export_group_norm_agrees_with_torch_group_norm() -> None:
    """The export-only shim must be the same function, not a different one."""

    from src.model.inference_export.blla.export import _ExportGroupNorm

    torch.manual_seed(0)
    reference = torch.nn.GroupNorm(32, 64).eval()
    torch.nn.init.normal_(reference.weight)
    torch.nn.init.normal_(reference.bias)
    staged = _ExportGroupNorm(reference).eval()

    values = torch.randn(1, 64, 30, 17)
    with torch.no_grad():
        expected = reference(values)
        actual = staged(values)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0)
    # The shim reuses the original module, so checkpoint keys are unchanged.
    assert set(staged.state_dict()) == {"layer.weight", "layer.bias"}


def test_export_swap_leaves_the_runtime_model_untouched() -> None:
    """The staged reduction is trace-only: the native oracle must not change."""

    from src.model.inference_export.blla.export import (
        _ExportGroupNorm,
        _with_export_group_norm,
    )
    from inference.architectures.blla.blla_model import _GroupNorm

    model = BLLATorchModel().eval()
    exportable = _with_export_group_norm(model)

    assert all(
        isinstance(getattr(exportable, name), _ExportGroupNorm)
        for name in ("Gn_1", "Gn_3", "Gn_5", "Gn_7", "Gn_9", "Gn_13")
    )
    assert all(
        isinstance(getattr(model, name), _GroupNorm)
        for name in ("Gn_1", "Gn_3", "Gn_5", "Gn_7", "Gn_9", "Gn_13")
    )

    values = torch.randn(1, 3, 1800, 24)
    with torch.no_grad():
        assert torch.allclose(exportable(values), model(values), atol=1e-4, rtol=0)


@pytest.mark.skipif(not BLLA_ONNX_ARTIFACT.is_file(), reason="BLLA ONNX artifact missing")
def test_exported_blla_graph_avoids_the_flat_group_reduction() -> None:
    """Guard the ONNX parity fix.

    ``nn.GroupNorm`` lowers to ``Reshape([0, 32, -1]) -> InstanceNormalization``,
    which makes onnxruntime reduce >2e6 float32 values in one serial accumulator
    and drifts the logits by up to 0.89 on a real page. If a future exporter
    change reintroduces that node, catch it here rather than in the ml-marked
    parity suite.
    """

    onnx = pytest.importorskip("onnx")

    graph = onnx.load(str(BLLA_ONNX_ARTIFACT)).graph
    op_types = [node.op_type for node in graph.node]

    assert "InstanceNormalization" not in op_types
    assert op_types.count("ReduceMean") == 24  # four staged reductions per GroupNorm
    assert op_types.count("LSTM") == 4  # still fused, not decomposed to Scan/Loop
