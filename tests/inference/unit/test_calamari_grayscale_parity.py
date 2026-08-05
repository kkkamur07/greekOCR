"""Training and serving must convert line images to grayscale identically.

The training pipeline (vendored Calamari, driven by ``src/train/calamari``) and the
serving pipeline (``inference/architectures/calamari/preprocessing``) each load line
images independently. When they disagree the model is fitted on one pixel
distribution and served another, which shows up only as a quiet accuracy loss.

These tests need no model weights, so they stay unmarked and run in the default CI
job — the ``ml`` marker is excluded from the Postgres integration job, which is
exactly why the original skew went unnoticed.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inference.architectures.calamari.preprocessing.pipeline import (
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


def test_grayscale_helper_is_the_only_convention_under_src() -> None:
    """No src/ module may reach for OpenCV's RGB->gray on a line image again.

    cvtColor is still legitimate for colour-space work that never feeds the model
    (mask/ROI geometry, BGR->RGB round trips), so this pins the *-2GRAY* families only.
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if "thirdparty" in path.parts:  # vendored ocrodeg / word-beam-search
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in ("COLOR_RGB2GRAY", "COLOR_BGR2GRAY", "COLOR_RGBA2GRAY"):
            if marker in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{marker}")

    # calamari_ocr/utils/image.py keeps the legacy branches so trainer params saved by
    # older runs still deserialize; center_normalizer's branch is unreachable at
    # channels=1 and measures geometry rather than producing model input.
    # trocr/augmentation/weather.py builds a luminance term to blend a snow effect
    # (`np.maximum(img, gray * 1.5 + 0.5)`) and hands back an RGB image. It is not
    # the train/serve conversion this test guards, so it is allowed here.
    #
    # Audit note, not fixed because src/ is audit-only: that same function then
    # does its actual grayscale with PIL's ImageOps.grayscale when isgray is set.
    # PIL and OpenCV use different luma coefficients, so an augmented training
    # image can be grayscaled by a different rule than the serving path uses -
    # exactly the skew this module exists to prevent, reached by a route the
    # marker list above cannot see.
    allowed = {
        "src/model/calamari/calamari_ocr/utils/image.py:COLOR_RGB2GRAY",
        "src/model/calamari/calamari_ocr/utils/image.py:COLOR_RGBA2GRAY",
        "src/model/calamari/calamari_ocr/ocr/dataset/imageprocessors/"
        "center_normalizer.py:COLOR_RGB2GRAY",
        "src/models/trocr/augmentation/weather.py:COLOR_RGB2GRAY",
    }
    assert set(offenders) <= allowed, f"new OpenCV grayscale conversion under src/: {offenders}"


def test_grayscale_module_has_no_training_only_dependencies() -> None:
    """The helper must stay importable without paiargparse/tfaip installed."""
    tree = ast.parse(_GRAYSCALE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert imported <= {"numpy", "PIL"}, f"unexpected dependency in grayscale helper: {imported}"


def test_serving_pipeline_still_reads_bytes_with_pil_convert_l() -> None:
    """Guard the other half of the contract: serving is the convention we matched."""
    pipeline_source = (
        REPO_ROOT / "inference/architectures/calamari/preprocessing/pipeline.py"
    ).read_text(encoding="utf-8")
    assert 'convert("L")' in pipeline_source


def test_helper_accepts_a_filesystem_path(tmp_path: Path) -> None:
    image_path = tmp_path / "line.png"
    image_path.write_bytes(_encode(_sample_rgb()))

    loaded = load_line_image_grayscale(image_path)

    assert loaded.ndim == 2
    assert loaded.dtype == np.uint8
