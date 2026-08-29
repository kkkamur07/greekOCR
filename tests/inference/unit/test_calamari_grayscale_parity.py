"""Training and serving must convert line images to grayscale identically.

The training pipeline (vendored Calamari, driven by ``src/train/calamari``) and the
serving pipeline (``nomikos_inference/architectures/calamari/preprocessing``) each load line
images independently. When they disagree the model is fitted on one pixel
distribution and served another, which shows up only as a quiet accuracy loss.

These tests need no model weights, so they stay unmarked and run in the default CI
job — the ``ml`` marker is excluded from the Postgres integration job, which is
exactly why the original skew went unnoticed.
"""

from __future__ import annotations

import importlib.util
import sys
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from nomikos_inference.architectures.calamari.preprocessing.pipeline import (
    preprocess_line_array_to_calamari_tensor,
    preprocess_line_image_bytes_to_calamari_tensor,
)
from tests.fixtures.paths import REPO_ROOT

# The vendored Calamari tree is not installed; it is put on PYTHONPATH by
# src/train/calamari/train_utils.py at training time. Its package __init__ pulls in
# paiargparse/tfaip, which only exist in the training environment, so load the
# dependency-free grayscale module straight from its file instead.
_GRAYSCALE_PATH = REPO_ROOT / "src/model/calamari/calamari_ocr/utils/grayscale.py"


def _load_training_grayscale():
    spec = importlib.util.spec_from_file_location("_calamari_grayscale", _GRAYSCALE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_line_image_grayscale


load_line_image_grayscale = _load_training_grayscale()


def _encode(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _sample_rgb() -> Image.Image:
    rng = np.random.default_rng(20260804)
    array = np.full((48, 240, 3), (222, 205, 170), dtype=np.uint8)  # parchment
    array = (array.astype(np.int16) + rng.integers(-12, 13, array.shape)).clip(0, 255)
    array = array.astype(np.uint8)
    for _ in range(40):  # iron-gall ink strokes
        x = int(rng.integers(0, 232))
        y = int(rng.integers(0, 30))
        array[y : y + 18, x : x + 8] = (60, 42, 30)
    return Image.fromarray(array, mode="RGB")


# Every PIL mode a scanned manuscript can plausibly arrive in. "P" and "CMYK" are the
# cases that used to diverge catastrophically: reading the raw array yields palette
# indices or ink values, not luminance.
def _mode_cases() -> list[tuple[str, bytes]]:
    rgb = _sample_rgb()
    return [
        ("RGB", _encode(rgb)),
        ("L", _encode(rgb.convert("L"))),
        ("P", _encode(rgb.convert("P", palette=Image.ADAPTIVE, colors=64))),
        ("RGBA", _encode(rgb.convert("RGBA"))),
        ("LA", _encode(rgb.convert("LA"))),
        ("1", _encode(rgb.convert("1"))),
        ("CMYK", _encode(rgb.convert("CMYK"), "TIFF")),
        ("JPEG", _encode(rgb, "JPEG")),
    ]


@pytest.mark.parametrize(("mode", "image_bytes"), _mode_cases(), ids=lambda value: value)
def test_training_grayscale_matches_serving_conversion(mode: str, image_bytes: bytes) -> None:
    training = load_line_image_grayscale(BytesIO(image_bytes))
    with Image.open(BytesIO(image_bytes)) as image:
        serving = np.asarray(image.convert("L"), dtype=np.uint8)

    assert training.dtype == np.uint8, mode
    assert training.shape == serving.shape, mode
    np.testing.assert_array_equal(training, serving, err_msg=f"grayscale skew for mode {mode}")


@pytest.mark.parametrize(("mode", "image_bytes"), _mode_cases(), ids=lambda value: value)
def test_training_and_serving_produce_the_same_model_input(mode: str, image_bytes: bytes) -> None:
    """End-to-end: the tensor handed to the model must not depend on which side built it."""
    from_training_load = preprocess_line_array_to_calamari_tensor(
        load_line_image_grayscale(BytesIO(image_bytes))
    )
    from_serving_bytes = preprocess_line_image_bytes_to_calamari_tensor(image_bytes)

    np.testing.assert_array_equal(
        from_training_load, from_serving_bytes, err_msg=f"train/serve skew for mode {mode}"
    )


# A `COLOR_*2GRAY` allowlist over the whole of `src/` used to stand here. It was
# deleted rather than extended: it scanned file *text*, so it passed on a comment and
# said nothing about which code path feeds the model, and it graded modules by location
# instead of by behaviour. It went red on `src/models/trocr/augmentation/weather.py`,
# whose grayscale is a luminance intermediate inside a fog composite that never becomes
# a model input tensor -- a false positive answerable only by growing the allowlist.
# Extending the same scan to `nomikos_inference/` would have needed another entry on
# identical terms (`architectures/calamari/preprocessing/geometry.py`), which
# measures geometry rather than producing model input.
#
# The property it was reaching for -- training and serving derive the same luminance
# from the same bytes -- is enforced above by
# `test_training_and_serving_produce_the_same_model_input`, which runs both real
# implementations over eight PIL modes and compares the tensors. That test fails on a
# real skew; the scan only failed on a spelling.


# `test_grayscale_module_has_no_training_only_dependencies` stood here and AST-scanned
# the vendored grayscale module for imports outside {numpy, PIL}. `_load_training_grayscale()`
# runs at module scope above, so a training-only import already errors every test in this
# file on collection -- the scan could only ever restate what import already proved.
#
# `test_serving_pipeline_still_reads_bytes_with_pil_convert_l` stood here and asserted
# the literal `convert("L")` appeared in the serving pipeline's source. A docstring
# mentioning it satisfied that, and a live cv2 fast path added above it did not break
# it. The executing parity test covers the same half of the contract by running the
# pipeline, so the grep was removed rather than kept as a second opinion that could only
# ever agree for the wrong reason.
