"""Guards for the BLLA ONNX export's staged GroupNorm reduction.

Restored by ADR 0006 with the exporter they guard. The shim has to be proven
equal to ``nn.GroupNorm`` on the full model at real page dimensions *and*
proven not to leak into the graph that gets traced for anything but export.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT

torch = pytest.importorskip("torch")

from src.model.inference_export.blla import export_blla_onnx  # noqa: E402
from src.model.inference_export.blla.model import BLLATorchModel  # noqa: E402

BLLA_CHECKPOINT = REPO_ROOT / "src/hf/staging/models/segmentation/blla/v1/stable/blla.safetensors"


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

    This shipped once: the ``blla.onnx`` published at revision ``444d51dd`` was
    exported before the staged reduction existed and carried six
    ``InstanceNormalization`` nodes, drifting logits by 1.5e-01 and
    restructuring the polygons of the two shortest lines on the page
    (``registry.yaml:33-40``).
    """

    onnx = pytest.importorskip("onnx")

    destination = tmp_path / "blla.onnx"
    export_blla_onnx(BLLA_CHECKPOINT, destination, example_width=64)
    op_types = [node.op_type for node in onnx.load(str(destination)).graph.node]

    assert "InstanceNormalization" not in op_types
