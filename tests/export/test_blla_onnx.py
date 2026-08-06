"""BLLA ONNX export and runtime tests.

Restored by ADR 0006, which puts the ONNX runtime back. They live under ``tests/export`` rather than
``tests/inference`` because they import Torch: the graph is the export-time
oracle, and nothing a researcher installs can import it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inference.architectures.blla.blla import run_blla_logits
from inference.architectures.blla.blla_decoder.common import resize_heatmaps_nearest
from inference.architectures.blla.blla_preprocessing import preprocess_blla_image
from tests.fixtures.paths import REPO_ROOT

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from src.model.inference_export.blla import export_blla_onnx  # noqa: E402
from src.model.inference_export.blla.model import BLLATorchModel  # noqa: E402

BLLA_ARTIFACT = REPO_ROOT / "src/hf/staging/models/segmentation/blla/v1/stable/blla.safetensors"


def test_the_numpy_decoder_head_matches_the_torch_one_on_a_real_page() -> None:
    """Resize *and* sigmoid together, on a real page's logits.

    ADR 0006 replaced ``interpolate`` + ``torch.sigmoid`` at the head of the
    decoder with NumPy, and that substitution is the one thing in the swap that
    changes arithmetic rather than moving it: the resize is exact index
    arithmetic, but `1/(1+exp(-x))` is not bit-identical to Torch's sigmoid
    kernel.

    The bound that matters is not "equal", it is "cannot flip a decision". The
    decoder thresholds probabilities at 0.17 and 0.5, so a disagreement of one
    float32 ULP is unobservable and a disagreement near either threshold is not.
    Random values would not test this - the real page is what puts probabilities
    *at* the boundary.
    """
    from inference.architectures.blla.blla_decoder import _sigmoid
    from tests.fixtures.paths import SEGMENT_PAGE

    with Image.open(SEGMENT_PAGE) as image:
        prepared = preprocess_blla_image(image.convert("RGB"))
    native = BLLATorchModel().eval()
    native.load_state_dict(load_file(BLLA_ARTIFACT, device="cpu"), strict=True)
    with torch.inference_mode():
        logits = native(torch.from_numpy(prepared.array).unsqueeze(0))[0].numpy()

    height, width = prepared.scaled_gray.shape
    with torch.inference_mode():
        expected = (
            torch.sigmoid(
                F.interpolate(torch.from_numpy(logits).unsqueeze(0), size=(height, width))[0]
            )
            .numpy()
            .astype(np.float32)
        )
    actual = _sigmoid(resize_heatmaps_nearest(logits, height=height, width=width))

    assert actual.shape == expected.shape
    # One ULP at float32 magnitude 1.0. Two orders of magnitude below the
    # distance any pixel would have to travel to cross 0.17 or 0.5.
    assert np.abs(actual - expected).max() <= 2e-07
    for threshold in (0.17, 0.5):
        assert np.array_equal(actual >= threshold, expected >= threshold)


@pytest.fixture(scope="module")
def blla_onnx_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not BLLA_ARTIFACT.is_file():
        pytest.skip("native BLLA checkpoint is unavailable")
    destination = tmp_path_factory.mktemp("blla-onnx") / "blla.onnx"
    export_blla_onnx(BLLA_ARTIFACT, destination, example_width=64)
    return destination


def test_blla_onnx_raw_logits_match_native_graph(blla_onnx_path: Path) -> None:
    native = BLLATorchModel().eval()
    native.load_state_dict(load_file(BLLA_ARTIFACT, device="cpu"), strict=True)
    inputs = np.random.default_rng(11).random((1, 3, 1800, 64), dtype=np.float32)

    with torch.inference_mode():
        native_logits = native(torch.from_numpy(inputs)).numpy()
    onnx_logits = run_blla_logits(inputs, model_path=blla_onnx_path)

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
    logits = run_blla_logits(inputs, model_path=blla_onnx_path)

    assert logits.shape == (1, 4, 450, 17)
    assert logits.shape == native_logits.shape
    np.testing.assert_allclose(logits, native_logits, rtol=2e-3, atol=1e-3)
