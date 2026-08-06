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
from inference.architectures.blla.blla_preprocessing import (
    MAX_WIDTH_TO_HEIGHT_RATIO,
    preprocess_blla_image,
)
from src.model.inference_export.blla import export_blla_onnx
from src.model.inference_export.blla.model import BLLATorchModel
from tests.fixtures.paths import REPO_ROOT, SEGMENT_PAGE

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

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


def _panorama_past_the_width_clamp() -> Image.Image:
    """The page fixture repeated into a scan far wider than the clamp allows.

    Noise would not exercise this. The ONNX/Torch gap being bounded here is a
    rounding difference in a reduction over the width axis, and it only shows up
    on spatially correlated post-ReLU activations - which is why ADR 0006
    measured the original defect on a page and not on ``rng.random``.
    """

    with Image.open(SEGMENT_PAGE) as page:
        page = page.convert("RGB")
        # Four times the clamp, so the clamp is unambiguously what sets the
        # width even if the fixture is ever replaced with a differently
        # proportioned page.
        target = 4 * MAX_WIDTH_TO_HEIGHT_RATIO * page.height
        tiles = -(-target // page.width)
        panorama = Image.new("RGB", (tiles * page.width, page.height))
        for index in range(tiles):
            panorama.paste(page, (index * page.width, 0))
        return panorama.crop((0, 0, target, page.height))


def test_blla_onnx_parity_survives_the_widest_input_the_clamp_admits(
    blla_onnx_path: Path,
) -> None:
    """Hold ONNX/Torch agreement at ``MAX_WIDTH_TO_HEIGHT_RATIO``'s own bound.

    The graph's agreement with the Torch oracle decays with the scaled width,
    and the width is the only free axis of the input. Measured on this fixture
    tiled to each width, ONNX against Torch on the raw logits:

        width    rms |d|   p99.9 |d|   max |d|   logits crossing sigmoid 0.5
         2471    1.7e-05     1.4e-04   1.5e-03   0     <- a real page, ADR 0006
         3600    2.2e-05     1.5e-04   1.1e-03   0
         5400    3.6e-05     2.5e-04   2.3e-03   0
         7200    7.5e-05     4.7e-04   1.2e-02   1
         9000    9.1e-05     6.1e-04   2.1e-02   1
        14400    1.9e-04     1.9e-03   2.4e-02   3

    ``MAX_WIDTH_TO_HEIGHT_RATIO`` used to be 8, which permitted the last row.
    The primary bound here is the RMS. The maximum is an extreme-value statistic
    that swings 5x between neighbouring widths on the same page, and the flip
    count is a single-digit integer that moves with the content - the squeezed
    panorama this test builds flips nothing at 5400, while a proportionally
    tiled source of a different ratio flips one. The RMS tracks the width almost
    linearly, so it is what catches the clamp being raised.

    All three assertions were checked against a raised clamp rather than assumed
    to bite: on this same panorama at ratio 8 they measure 1.16e-04, 1.47e-02 and
    4 crossings, and each exceeds its bound.

    The width is read from the constant rather than written out, so raising the
    clamp moves this test with it instead of leaving it guarding a width nothing
    produces any more.
    """

    native = BLLATorchModel().eval()
    native.load_state_dict(load_file(BLLA_ARTIFACT, device="cpu"), strict=True)
    prepared = preprocess_blla_image(_panorama_past_the_width_clamp())
    inputs = prepared.array[None, ...]
    assert inputs.shape[3] == BLLATorchModel.input_height * MAX_WIDTH_TO_HEIGHT_RATIO

    with torch.inference_mode():
        native_logits = native(torch.from_numpy(inputs)).numpy()
    onnx_logits = run_blla_logits(inputs, model_path=blla_onnx_path)

    difference = np.abs(onnx_logits - native_logits)
    # Measured 4.2e-05 and 5.2e-03 on this input; a real page at width 2471
    # measures 1.7e-05 and 1.5e-03.
    assert np.sqrt(np.mean(np.square(difference, dtype=np.float64))) <= 1e-04
    assert difference.max() <= 1e-02
    # The decoder thresholds the region channel at 0.5 and is discontinuous
    # there, so a logit that changes sign is a segmentation change rather than a
    # rounding difference. None do on this input at this width.
    assert np.array_equal(onnx_logits >= 0.0, native_logits >= 0.0)


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
