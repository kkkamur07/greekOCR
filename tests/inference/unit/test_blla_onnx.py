"""Focused ONNX BLLA export and Torch-free runtime tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from inference.architectures.blla.blla_model import BLLATorchModel
from inference.architectures.blla.onnx import run_blla_onnx_logits
from inference.architectures.blla.blla_preprocessing import preprocess_blla_image_numpy
from inference.architectures.blla.blla_decoder.common import resize_heatmaps_nearest
from safetensors.torch import load_file
from src.model.inference_export.blla import export_blla_onnx

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

REPO_ROOT = Path(__file__).resolve().parents[3]
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


def test_blla_numpy_preprocessing_is_float32() -> None:
    image = Image.new("RGB", (20, 10), (0, 64, 255))

    prepared = preprocess_blla_image_numpy(image)

    assert prepared.array.dtype == np.float32


def test_blla_numpy_interpolation_matches_torch_nearest() -> None:
    values = np.random.default_rng(7).random((4, 9, 13), dtype=np.float32)

    actual = resize_heatmaps_nearest(values, height=31, width=17)
    expected = F.interpolate(torch.from_numpy(values[None]), size=(31, 17))[0].numpy()

    np.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope="module")
def blla_onnx_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not BLLA_ARTIFACT.is_file():
        pytest.skip("native BLLA checkpoint is unavailable")
    destination = tmp_path_factory.mktemp("blla-onnx") / "blla.onnx"
    export_blla_onnx(BLLA_ARTIFACT, destination, example_width=64)
    return destination


def test_blla_onnx_embeds_dynamic_width_metadata(blla_onnx_path: Path) -> None:
    import onnx

    model = onnx.load(str(blla_onnx_path))
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    input_shape = model.graph.input[0].type.tensor_type.shape
    width_dimension = input_shape.dim[3]

    assert metadata["format"] == "blla-onnx-v1"
    assert metadata["graph"] == "inference-owned-blla-torch-v1"
    assert width_dimension.dim_param == "width"


def test_blla_onnx_raw_logits_match_native_graph(blla_onnx_path: Path) -> None:
    native = BLLATorchModel().eval()
    native.load_state_dict(load_file(BLLA_ARTIFACT, device="cpu"), strict=True)
    inputs = np.random.default_rng(11).random((1, 3, 1800, 64), dtype=np.float32)

    with torch.inference_mode():
        native_logits = native(torch.from_numpy(inputs)).numpy()
    onnx_logits = run_blla_onnx_logits(inputs, model_path=blla_onnx_path)

    assert onnx_logits.shape == native_logits.shape
    np.testing.assert_allclose(onnx_logits, native_logits, rtol=2e-3, atol=1e-3)


def test_blla_onnx_accepts_a_second_dynamic_width(blla_onnx_path: Path) -> None:
    """Check parity at a width other than the trace width.

    A shape silently constant-folded during export would pass the trace-width
    parity test but produce wrong values (not just wrong shapes) here.
    """
    native = BLLATorchModel().eval()
    native.load_state_dict(load_file(BLLA_ARTIFACT, device="cpu"), strict=True)
    inputs = np.random.default_rng(12).random((1, 3, 1800, 65), dtype=np.float32)

    with torch.inference_mode():
        native_logits = native(torch.from_numpy(inputs)).numpy()
    logits = run_blla_onnx_logits(inputs, model_path=blla_onnx_path)

    assert logits.shape == (1, 4, 450, 17)
    assert logits.shape == native_logits.shape
    np.testing.assert_allclose(logits, native_logits, rtol=2e-3, atol=1e-3)
