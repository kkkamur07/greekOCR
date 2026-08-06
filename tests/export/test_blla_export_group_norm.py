"""Guards for the BLLA ONNX export's staged GroupNorm reduction.

Restored by ADR 0006 with the exporter they guard. The shim has to be proven
equal to ``nn.GroupNorm`` *and* proven not to leak into the graph that gets
traced for anything but export - which is why it is asserted against the
**published** artifact, not only against a freshly exported one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.model.inference_export.blla import export_blla_onnx
from src.model.inference_export.blla.model import BLLATorchModel
from tests.fixtures.paths import REPO_ROOT

# The published artifact, as fetched from the Hub revision the registry pins.
BLLA_ONNX_ARTIFACT = REPO_ROOT / "src/hf/cache/blla-segment/stable/blla.onnx"
BLLA_CHECKPOINT = REPO_ROOT / "src/hf/staging/models/segmentation/blla/v1/stable/blla.safetensors"


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
    from src.model.inference_export.blla.model import _GroupNorm

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


@pytest.mark.skipif(not BLLA_CHECKPOINT.is_file(), reason="native BLLA checkpoint is unavailable")
def test_exported_blla_graph_avoids_the_flat_group_reduction(tmp_path: Path) -> None:
    """Guard the ONNX parity fix, on a graph this exporter just produced.

    ``nn.GroupNorm`` lowers to ``Reshape([0, 32, -1]) -> InstanceNormalization``,
    which makes onnxruntime reduce >2e6 float32 values in one serial accumulator
    and drifts the logits by up to 0.89 on a real page. If a future exporter
    change reintroduces that node, catch it here.
    """

    onnx = pytest.importorskip("onnx")

    destination = tmp_path / "blla.onnx"
    export_blla_onnx(BLLA_CHECKPOINT, destination, example_width=64)
    op_types = [node.op_type for node in onnx.load(str(destination)).graph.node]

    assert "InstanceNormalization" not in op_types
    assert op_types.count("ReduceMean") == 24  # four staged reductions per GroupNorm
    assert op_types.count("LSTM") == 4  # still fused, not decomposed to Scan/Loop


@pytest.mark.skipif(not BLLA_ONNX_ARTIFACT.is_file(), reason="BLLA ONNX artifact missing")
def test_the_published_blla_artifact_is_the_fixed_export() -> None:
    """The artifact the registry pins must be the graph the exporter produces today.

    It was not, until 2026-08-05. The ``blla.onnx`` published at revision
    ``444d51dd`` carried six ``InstanceNormalization`` nodes and no
    ``ReduceMean``: it had been exported before the staged reduction existed, so
    ADR 0004 retired the ONNX runtime while the *published* artifact still had
    the defect the fix was written for. Measured against the Torch graph on
    ``segment_page.jpeg``:

        444d51dd:   logits max |d| 1.5e-01, 33/34 baselines identical, min IoU 0.73
        5c20a584:   logits max |d| 1.5e-03, 34/34 baselines identical, min IoU 1.00

    The drift landed exactly where it was predicted to - the two shortest lines
    on the page, whose polygons restructure when a handful of pixels cross the
    0.5 boundary. Transcription was never affected; this is segmentation
    geometry.

    Keeping the check pointed at the *published* file rather than at a fresh
    export is the whole point: an exporter that is correct and an artifact that
    is correct are different claims, and it was the second one that was false.
    """

    onnx = pytest.importorskip("onnx")

    op_types = [node.op_type for node in onnx.load(str(BLLA_ONNX_ARTIFACT)).graph.node]

    assert "InstanceNormalization" not in op_types
    assert op_types.count("ReduceMean") == 24
