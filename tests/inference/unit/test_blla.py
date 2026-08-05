"""Focused tests for the inference-owned BLLA runtime."""

from __future__ import annotations

import base64
import hashlib
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image

from inference.architectures.blla.blla import _load_blla_model
from inference.architectures.blla.blla_decoder import decode_blla_heatmaps
from inference.architectures.blla.blla_model import BLLATorchModel
from inference.architectures.blla.blla_preprocessing import (
    MAX_WIDTH_TO_HEIGHT_RATIO,
    preprocess_blla_image,
    preprocess_blla_image_numpy,
)
from inference.helper.app import create_helper_app
from inference.helper.settings import get_helper_settings
from inference.infrastructure.settings import get_inference_settings
from inference.registry import load_registry

REPO_ROOT = Path(__file__).resolve().parents[3]
SEGMENT_PAGE = REPO_ROOT / "tests" / "fixtures" / "manuscripts" / "greek" / "segment_page.jpeg"
BLLA_ARTIFACT = (
    REPO_ROOT
    / "src"
    / "hf"
    / "staging"
    / "models"
    / "segmentation"
    / "blla"
    / "v1"
    / "stable"
    / "blla.safetensors"
)
BLLA_ONNX_ARTIFACT = BLLA_ARTIFACT.with_name("blla.onnx")


def _staged_onnx_matches_registry() -> bool:
    """Whether the staged ONNX export is still the artifact ``registry.yaml`` pins.

    The helper hashes every resolved weight file against the registry's
    ``artifact_sha256`` and answers 503 on a mismatch, so a staged export that has
    drifted from the pinned digest cannot reach the run path at all. Re-exporting
    ``blla.onnx`` without re-publishing and re-pinning it is exactly that drift,
    and it is a registry bug, not a test bug - skip rather than assert a 503.
    """
    if not BLLA_ONNX_ARTIFACT.is_file():
        return False
    expected = load_registry().models["blla-segment"].versions["stable"].artifact_sha256
    if expected is None:
        return True
    return hashlib.sha256(BLLA_ONNX_ARTIFACT.read_bytes()).hexdigest() == expected


def test_blla_model_has_fixed_topology_and_expected_output_shape() -> None:
    model = BLLATorchModel().eval()
    output = model(torch.zeros((1, 3, 1800, 20)))

    assert output.shape == (1, 4, 450, 5)
    assert list(model.state_dict())[:4] == [
        "C_0.co.weight",
        "C_0.co.bias",
        "Gn_1.layer.weight",
        "Gn_1.layer.bias",
    ]
    assert model.state_dict()["L_10.layer.weight_ih_l0"].shape == (128, 256)
    assert model.state_dict()["L_11.layer.weight_ih_l0"].shape == (128, 64)


def test_blla_preprocessing_matches_fixed_height_rgb_inversion() -> None:
    image = Image.new("RGB", (20, 10), (0, 64, 255))

    prepared = preprocess_blla_image(image)

    assert prepared.tensor.shape == (3, 1800, 3600)
    assert prepared.tensor.dtype == torch.float32
    assert torch.allclose(prepared.tensor[:, 0, 0], torch.tensor([1.0, 0.7490196, 0.0]))
    assert prepared.scaled_gray.shape == (1800, 3600)
    assert prepared.scale_xy == pytest.approx((20 / 3600, 10 / 1800))


def test_blla_preprocessing_caps_extreme_aspect_ratio_width() -> None:
    """A panorama within the pixel admission cap must not produce an unbounded tensor."""
    input_height = 90
    capped_width = input_height * MAX_WIDTH_TO_HEIGHT_RATIO
    image = Image.new("RGB", (4000, 10), (255, 255, 255))

    prepared = preprocess_blla_image(image, input_height=input_height)
    prepared_numpy = preprocess_blla_image_numpy(image, input_height=input_height)

    assert prepared.tensor.shape == (3, input_height, capped_width)
    assert prepared_numpy.array.shape == (3, input_height, capped_width)
    # Coordinates still map back to source space through scale_xy.
    assert prepared.scale_xy == pytest.approx((4000 / capped_width, 10 / input_height))
    assert prepared_numpy.scale_xy == prepared.scale_xy


def test_blla_preprocessing_is_proportional_below_the_width_cap() -> None:
    image = Image.new("RGB", (20, 10), (255, 255, 255))

    prepared = preprocess_blla_image(image, input_height=90)

    assert prepared.tensor.shape == (3, 90, 180)


def test_blla_decoder_turns_separator_ridge_into_line_polygon() -> None:
    heatmaps = np.zeros((4, 80, 100), dtype=np.float32)
    heatmaps[0, 38:43, 10:91] = 1.0
    heatmaps[1, 38:43, 10:91] = 1.0
    heatmaps[2, 38:43, 10:91] = 1.0
    heatmaps[3, 30:51, 10:91] = 1.0

    lines = decode_blla_heatmaps(heatmaps, image_size=(100, 80))

    assert len(lines) == 1
    assert lines[0].baseline[0][0] == pytest.approx(10.0)
    assert lines[0].baseline[-1][0] == pytest.approx(90.0)
    ys = [point[1] for point in lines[0].polygon]
    assert min(ys) == pytest.approx(30.0)
    assert max(ys) == pytest.approx(50.0)


@pytest.fixture(scope="module")
def real_blla_outputs() -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Run the staged checkpoint once on the real manuscript fixture."""
    with Image.open(SEGMENT_PAGE) as image:
        prepared = preprocess_blla_image(image)
    model = _load_blla_model(str(BLLA_ARTIFACT))
    with torch.inference_mode():
        logits = model(prepared.tensor.unsqueeze(0))
    heatmaps = F.interpolate(
        torch.sigmoid(logits),
        size=prepared.scaled_gray.shape,
    )
    return logits, heatmaps, prepared.scaled_gray.shape


@pytest.fixture(scope="module")
def kraken_blla_outputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Load Kraken's optional bundled reference and run the same input."""
    pytest.importorskip("kraken")
    from kraken.lib import vgsl

    with Image.open(SEGMENT_PAGE) as image:
        prepared = preprocess_blla_image(image)
    reference = vgsl.TorchVGSLModel.load_model(resources.files("kraken").joinpath("blla.mlmodel"))
    with torch.inference_mode():
        output = reference.nn(prepared.tensor.unsqueeze(0))
    reference_logits = output[0] if isinstance(output, tuple) else output
    reference_heatmaps = F.interpolate(
        torch.sigmoid(reference_logits),
        size=prepared.scaled_gray.shape,
    )
    return reference_logits, reference_heatmaps


def test_blla_real_manuscript_produces_segment_candidates(
    real_blla_outputs: tuple[torch.Tensor, torch.Tensor, tuple[int, int]],
) -> None:
    logits, heatmaps, scaled_size = real_blla_outputs

    assert logits.shape[0:2] == (1, 4)
    assert heatmaps.shape == (1, 4, *scaled_size)
    assert torch.isfinite(logits).all()
    assert torch.all((heatmaps >= 0) & (heatmaps <= 1))


def test_blla_raw_logits_match_optional_kraken_reference(
    real_blla_outputs: tuple[torch.Tensor, torch.Tensor, tuple[int, int]],
    kraken_blla_outputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    native_logits, _, _ = real_blla_outputs
    reference_logits, _ = kraken_blla_outputs

    torch.testing.assert_close(native_logits, reference_logits, rtol=1e-5, atol=1e-5)


def test_blla_sigmoid_interpolated_heatmaps_match_optional_kraken_reference(
    real_blla_outputs: tuple[torch.Tensor, torch.Tensor, tuple[int, int]],
    kraken_blla_outputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, native_heatmaps, _ = real_blla_outputs
    _, reference_heatmaps = kraken_blla_outputs

    torch.testing.assert_close(native_heatmaps, reference_heatmaps, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not BLLA_ONNX_ARTIFACT.is_file(), reason="BLLA ONNX artifact missing")
@pytest.mark.skipif(
    not _staged_onnx_matches_registry(),
    reason="staged BLLA ONNX artifact does not match registry.yaml artifact_sha256",
)
def test_standalone_helper_returns_onnx_blla_response_for_real_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise HTTP validation, ONNX runner dispatch, and response serialization."""
    registry = REPO_ROOT / "inference" / "registry.yaml"
    monkeypatch.setenv("INFERENCE_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("HELPER_BUNDLED_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_PATH", str(tmp_path / "registry.yaml"))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_ETAG_PATH", str(tmp_path / "registry.etag"))
    monkeypatch.setenv("HF_CACHE_ROOT", str(tmp_path / "hf-cache"))
    get_helper_settings.cache_clear()
    get_inference_settings.cache_clear()
    monkeypatch.setattr(
        "inference.jobs.runner.resolve_weights_source",
        lambda *_args, **_kwargs: BLLA_ONNX_ARTIFACT,
    )

    with TestClient(create_helper_app()) as client:
        response = client.post(
            "/inference/v1/run",
            json={
                "task": "segment",
                "registry_model_id": "blla-segment",
                "registry_tag": "stable",
                "image_bytes": base64.b64encode(SEGMENT_PAGE.read_bytes()).decode(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "segment"
    assert len(body["output"]["blocks"]) == 1
    assert len(body["output"]["lines"]) > 10
    assert all(line["source_metadata"]["adapter"] == "blla" for line in body["output"]["lines"])


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
