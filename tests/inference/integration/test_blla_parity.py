"""Inference-only parity checks against Kraken's bundled BLLA oracle."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from shapely.geometry import LineString, Polygon

pytestmark = pytest.mark.ml
pytest.importorskip("kraken")
pytest.importorskip("onnxruntime")

from kraken import blla  # noqa: E402
from kraken.lib import vgsl  # noqa: E402
from kraken.lib.dataset import ImageInputTransforms  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from inference.architectures.blla.blla import run_blla_segment  # noqa: E402
from inference.architectures.blla.blla_decoder import decode_blla_heatmaps  # noqa: E402
from inference.architectures.blla.onnx import (  # noqa: E402
    run_blla_onnx_logits,
    run_blla_onnx_segment,
)
from inference.architectures.blla.blla_model import BLLATorchModel  # noqa: E402
from inference.architectures.blla.blla_preprocessing import (  # noqa: E402
    preprocess_blla_image,
)
from inference.preprocessing.segment_geometry import simplify_blla_boundary  # noqa: E402
from tests.fixtures.paths import REPO_ROOT, SEGMENT_PAGE  # noqa: E402


NATIVE_CHECKPOINT = (
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
ONNX_CHECKPOINT = NATIVE_CHECKPOINT.with_name("blla.onnx")
MANUSCRIPT_PAGE_ROOT = REPO_ROOT / "nomicous" / "data" / "manuscripts" / "pages"
MANUSCRIPT_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
POLYGON_MIN_IOU = 0.90
POLYGON_MAX_HAUSDORFF_PX = 32.0
ONNX_MEAN_LOGIT_MAX = 0.002
ONNX_P99_LOGIT_MAX = 0.02
ONNX_MAX_LOGIT_MAX = 0.2
EXPECTED_MANUSCRIPT_SEGMENT_COUNTS = {
    "00000003.tif": 136,
    "0216_SMMJ_00036__221.jpg": 147,
    "Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_6.jpeg": 34,
    "Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_7.jpeg": 56,
    "Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_8.jpeg": 55,
    "renameee.jpeg": 56,
}

KRAKEN_CHECKPOINT = Path(str(resources.files("kraken").joinpath("blla.mlmodel")))
if not KRAKEN_CHECKPOINT.is_file():
    pytest.skip("Kraken blla.mlmodel oracle is unavailable", allow_module_level=True)
if not NATIVE_CHECKPOINT.is_file():
    pytest.skip("native BLLA safetensors checkpoint is unavailable", allow_module_level=True)
if not ONNX_CHECKPOINT.is_file():
    pytest.skip("BLLA ONNX checkpoint is unavailable", allow_module_level=True)


def _oracle_input(oracle: object, image: Image.Image) -> torch.Tensor:
    batch, channels, height, width = oracle.input  # type: ignore[attr-defined]
    padding = oracle.user_metadata.get("hyper_params", {}).get("padding", (0, 0))  # type: ignore[attr-defined]

    if isinstance(padding, int):
        padding = (padding,) * 4
    elif len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])

    return ImageInputTransforms(
        batch,
        height,
        width,
        channels,
        padding,
        valid_norm=False,
    )(image)


def _line_points(line: object, attribute: str) -> list[list[float]]:
    points = getattr(line, attribute)
    return [[float(x), float(y)] for x, y in points]


def _assert_native_ids_and_metadata(
    native_result: object,
    *,
    expect_sequential_raw_order: bool,
    image_size: tuple[int, int] | None = None,
) -> None:
    lines = native_result.lines  # type: ignore[attr-defined]
    blocks = native_result.blocks  # type: ignore[attr-defined]

    assert len(blocks) == 1
    assert blocks[0].external_id == "blla-block-1"
    assert blocks[0].order == 0

    if image_size is not None:
        width, height = image_size
        assert blocks[0].box["points"] == [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)],
        ]
    assert len({line.external_id for line in lines}) == len(lines)
    assert [line.order for line in lines] == list(range(len(lines)))
    assert all(line.block_external_id == "blla-block-1" for line in lines)
    assert all(line.source_metadata["adapter"] == "blla" for line in lines)
    assert all(line.source_metadata["decoder"] == "native" for line in lines)
    assert all(line.source_metadata["raw_order"] >= 0 for line in lines)

    if expect_sequential_raw_order:
        assert [line.source_metadata["raw_order"] for line in lines] == list(range(len(lines)))

    assert all(line.kraken_ceiling for line in lines)


def _assert_otsu_preserves_ceilings(off_result: object, on_result: object) -> None:
    off_ceilings = [line.kraken_ceiling for line in off_result.lines]  # type: ignore[attr-defined]
    on_ceilings = [line.kraken_ceiling for line in on_result.lines]  # type: ignore[attr-defined]
    assert off_ceilings
    assert on_ceilings
    for ceiling in off_ceilings:
        assert any(
            ceiling is not None
            and candidate is not None
            and _polygon_similarity(ceiling, candidate)[0] >= POLYGON_MIN_IOU
            and _polygon_similarity(ceiling, candidate)[1] <= POLYGON_MAX_HAUSDORFF_PX
            for candidate in on_ceilings
        )
        _assert_native_ids_and_metadata(on_result, expect_sequential_raw_order=False)


def _polygon_similarity(
    expected: list[list[float]],
    actual: list[list[float]],
) -> tuple[float, float]:
    expected_polygon = Polygon(expected)
    actual_polygon = Polygon(actual)

    if not expected_polygon.is_valid:
        expected_polygon = expected_polygon.buffer(0)
    if not actual_polygon.is_valid:
        actual_polygon = actual_polygon.buffer(0)

    union_area = expected_polygon.union(actual_polygon).area
    intersection_area = expected_polygon.intersection(actual_polygon).area
    iou = intersection_area / union_area if union_area else 0.0

    return iou, expected_polygon.hausdorff_distance(actual_polygon)


def _assert_native_matches_kraken_lines(
    oracle_lines: list[object],
    native_result: object,
    *,
    page_name: str,
) -> list[str]:
    native_lines = native_result.lines  # type: ignore[attr-defined]

    assert len(native_lines) == len(oracle_lines)
    _assert_native_ids_and_metadata(native_result, expect_sequential_raw_order=True)

    geometry_failures: list[str] = []
    for index, (oracle_line, native_line) in enumerate(zip(oracle_lines, native_lines)):
        oracle_baseline = _line_points(oracle_line, "baseline")
        oracle_boundary = _line_points(oracle_line, "boundary")

        assert native_line.baseline["points"] == oracle_baseline
        native_ceiling = native_line.kraken_ceiling
        assert native_ceiling is not None

        iou, hausdorff_distance = _polygon_similarity(oracle_boundary, native_ceiling)
        if iou < POLYGON_MIN_IOU or hausdorff_distance > POLYGON_MAX_HAUSDORFF_PX:
            geometry_failures.append(
                f"{page_name} line {index}: IoU={iou:.6f}, "
                f"Hausdorff={hausdorff_distance:.2f}px"
            )
        expected_points, _ = simplify_blla_boundary(native_ceiling)
        assert native_line.points == expected_points
        assert native_line.points == native_line.mask["points"]
    return geometry_failures


def test_native_blla_matches_kraken_on_real_fixture() -> None:
    """Compare preprocessing, weights, logits, ordering, and line count."""

    oracle = vgsl.TorchVGSLModel.load_model(KRAKEN_CHECKPOINT)
    image = Image.open(SEGMENT_PAGE).convert("RGB")

    oracle_input = _oracle_input(oracle, image)
    prepared = preprocess_blla_image(image, input_height=oracle.input[2])

    assert oracle_input.shape == prepared.tensor.shape
    assert torch.allclose(oracle_input, prepared.tensor, atol=1e-6, rtol=0)
    assert np.array_equal(
        prepared.scaled_gray,
        np.asarray(
            image.resize(
                (prepared.tensor.shape[2], prepared.tensor.shape[1]),
                Image.Resampling.LANCZOS,
            ).convert("L"),
            dtype=np.uint8,
        ),
    )

    native = BLLATorchModel().eval()
    native.load_state_dict(load_file(NATIVE_CHECKPOINT, device="cpu"), strict=True)

    oracle_state = oracle.nn.state_dict()
    native_state = native.state_dict()

    assert native_state.keys() == oracle_state.keys()
    assert all(torch.equal(native_state[name], oracle_state[name]) for name in native_state)

    with torch.inference_mode():
        oracle_logits, _ = oracle.nn(oracle_input.unsqueeze(0))
        native_logits = native(prepared.tensor.unsqueeze(0))

    logit_difference = native_logits - oracle_logits

    # Asserting the mean logit difference. 
    assert float(logit_difference.abs().mean()) < 1e-5
    assert float(logit_difference.abs().max()) < 1e-4

    oracle_result = blla.segment(image, model=oracle, raise_on_error=True)
    native_result = run_blla_segment(
        SEGMENT_PAGE.read_bytes(),
        model_path=NATIVE_CHECKPOINT,
    )
    assert len(oracle_result.lines) == 34
    assert len(native_result.lines) == len(oracle_result.lines)
    assert [line.baseline for line in oracle_result.lines] == [
        line.baseline["points"] for line in native_result.lines
    ]


def _assert_onnx_matches_kraken_page(
    oracle: object,
    page_path: Path,
) -> tuple[float, float, int]:
    with Image.open(page_path) as opened:
        image = opened.convert("RGB")
    oracle_input = _oracle_input(oracle, image)
    prepared = preprocess_blla_image(image, input_height=oracle.input[2])  # type: ignore[attr-defined]

    with torch.inference_mode():
        oracle_logits, _ = oracle.nn(oracle_input.unsqueeze(0))  # type: ignore[attr-defined]
    onnx_logits = run_blla_onnx_logits(
        prepared.tensor.unsqueeze(0).numpy(),
        model_path=ONNX_CHECKPOINT,
    )
    logit_difference = np.abs(onnx_logits - oracle_logits.numpy())
    mean_difference = float(logit_difference.mean())
    max_difference = float(logit_difference.max())
    p99_difference = float(np.percentile(logit_difference, 99))
    logit_failures = []
    if mean_difference >= ONNX_MEAN_LOGIT_MAX:
        logit_failures.append(f"mean={mean_difference:.8f}")
    if p99_difference >= ONNX_P99_LOGIT_MAX:
        logit_failures.append(f"p99={p99_difference:.8f}")
    if max_difference >= ONNX_MAX_LOGIT_MAX:
        logit_failures.append(f"max={max_difference:.8f}")
    assert not logit_failures, f"{page_path.name}: " + ", ".join(logit_failures)

    oracle_result = blla.segment(image, model=oracle, raise_on_error=True)
    onnx_result = run_blla_onnx_segment(
        page_path.read_bytes(),
        model_path=ONNX_CHECKPOINT,
    )
    expected_segment_count = EXPECTED_MANUSCRIPT_SEGMENT_COUNTS.get(page_path.name)
    if expected_segment_count is not None:
        assert len(oracle_result.lines) == expected_segment_count, page_path.name
    assert len(onnx_result.lines) == len(oracle_result.lines), page_path.name
    assert [line.order for line in onnx_result.lines] == list(range(len(onnx_result.lines)))
    assert all(line.kraken_ceiling for line in onnx_result.lines)

    similarities: list[tuple[float, float]] = []
    for oracle_line, onnx_line in zip(oracle_result.lines, onnx_result.lines, strict=True):
        baseline_distance = LineString(oracle_line.baseline).hausdorff_distance(
            LineString(onnx_line.baseline["points"])
        )
        assert baseline_distance <= POLYGON_MAX_HAUSDORFF_PX
        similarities.append(
            _polygon_similarity(
                _line_points(oracle_line, "boundary"),
                onnx_line.kraken_ceiling,
            )
        )

    assert similarities
    minimum_iou = min(iou for iou, _ in similarities)
    mean_iou = sum(iou for iou, _ in similarities) / len(similarities)
    maximum_hausdorff = max(hausdorff for _, hausdorff in similarities)
    assert minimum_iou >= POLYGON_MIN_IOU, (
        f"{page_path.name}: minimum IoU={minimum_iou:.6f}"
    )
    assert mean_iou >= 0.95, f"{page_path.name}: mean IoU={mean_iou:.6f}"
    assert maximum_hausdorff <= POLYGON_MAX_HAUSDORFF_PX, (
        f"{page_path.name}: maximum Hausdorff={maximum_hausdorff:.2f}px"
    )
    return mean_difference, max_difference, len(onnx_result.lines)


def test_onnx_blla_matches_kraken_on_real_fixture() -> None:
    """Compare the published ONNX artifact directly with Kraken's oracle."""

    oracle = vgsl.TorchVGSLModel.load_model(KRAKEN_CHECKPOINT)
    mean_difference, max_difference, segment_count = _assert_onnx_matches_kraken_page(
        oracle,
        SEGMENT_PAGE,
    )
    assert mean_difference < ONNX_MEAN_LOGIT_MAX
    assert max_difference < ONNX_MAX_LOGIT_MAX
    assert segment_count == 34


def test_torch_and_numpy_decoders_match_on_identical_real_logits() -> None:
    """Prove the Torch-free decoder is not the source of ONNX drift."""

    oracle = vgsl.TorchVGSLModel.load_model(KRAKEN_CHECKPOINT)
    with Image.open(SEGMENT_PAGE) as opened:
        image = opened.convert("RGB")
    prepared = preprocess_blla_image(image, input_height=oracle.input[2])
    with torch.inference_mode():
        logits, _ = oracle.nn(_oracle_input(oracle, image).unsqueeze(0))

    decoded = [
        decode_blla_heatmaps(
            logits[0].numpy(),
            image_size=image.size,
            threshold=0.17,
            raw_logits=True,
            scaled_gray=prepared.scaled_gray,
            torch_free=torch_free,
        )
        for torch_free in (False, True)
    ]
    assert [
        (line.baseline, line.polygon) for line in decoded[0]
    ] == [
        (line.baseline, line.polygon) for line in decoded[1]
    ]


def _manuscript_page_paths() -> list[Path]:
    if not MANUSCRIPT_PAGE_ROOT.is_dir():
        return []
    return sorted(
        path
        for path in MANUSCRIPT_PAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in MANUSCRIPT_SUFFIXES
    )


def test_kraken_native_and_onnx_match_on_all_manuscript_pages() -> None:
    """Compare logits, decoding, and contracts across every manuscript page."""

    page_paths = _manuscript_page_paths()
    if not page_paths:
        pytest.skip("no local manuscript pages are available")

    oracle = vgsl.TorchVGSLModel.load_model(KRAKEN_CHECKPOINT)
    native = BLLATorchModel().eval()

    native.load_state_dict(load_file(NATIVE_CHECKPOINT, device="cpu"), strict=True)
    oracle_state = oracle.nn.state_dict()
    native_state = native.state_dict()

    # Testing whether the graph of the model matches or not. 
    assert native_state.keys() == oracle_state.keys()
    assert all(torch.equal(native_state[name], oracle_state[name]) for name in native_state)

    onnx_failures: list[str] = []
    for page_path in page_paths:
        with Image.open(page_path) as opened:
            image = opened.convert("RGB")
        oracle_input = _oracle_input(oracle, image)
        prepared = preprocess_blla_image(image, input_height=oracle.input[2])
        
        assert oracle_input.shape == prepared.tensor.shape
        torch.testing.assert_close(oracle_input, prepared.tensor, rtol=0, atol=1e-6)
        
        assert np.array_equal(
            prepared.scaled_gray,
            np.asarray(
                image.resize(
                    (prepared.tensor.shape[2], prepared.tensor.shape[1]),
                    Image.Resampling.LANCZOS,
                ).convert("L"),
                dtype=np.uint8,
            ),
        )
        assert prepared.scale_xy == pytest.approx(
            (image.width / prepared.tensor.shape[2], image.height / prepared.tensor.shape[1])
        )

        with torch.inference_mode():
            oracle_logits, _ = oracle.nn(oracle_input.unsqueeze(0))
            native_logits = native(prepared.tensor.unsqueeze(0))

        logit_difference = native_logits - oracle_logits
        assert float(logit_difference.abs().mean()) < 1e-5, page_path.name
        assert float(logit_difference.abs().max()) < 1e-4, page_path.name

        oracle_heatmaps = F.interpolate(
            torch.sigmoid(oracle_logits),
            size=prepared.scaled_gray.shape,
        )
        native_heatmaps = F.interpolate(
            torch.sigmoid(native_logits),
            size=prepared.scaled_gray.shape,
        )
        torch.testing.assert_close(native_heatmaps, oracle_heatmaps, rtol=1e-5, atol=1e-5)

        oracle_result = blla.segment(image, model=oracle, raise_on_error=True)
        native_result = run_blla_segment(
            page_path.read_bytes(),
            model_path=NATIVE_CHECKPOINT,
        )
        assert not _assert_native_matches_kraken_lines(
            oracle_result.lines,
            native_result,
            page_name=page_path.name,
        )
        _assert_native_ids_and_metadata(
            native_result,
            expect_sequential_raw_order=True,
            image_size=image.size,
        )

        try:
            _assert_onnx_matches_kraken_page(oracle, page_path)
        except AssertionError as error:
            onnx_failures.append(str(error))

    assert not onnx_failures, "\n".join(onnx_failures)
