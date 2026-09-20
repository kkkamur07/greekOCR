"""PP-OCR recognition adapter: metadata, decode, isolation and dispatch.

Torch-free by construction, like everything under ``tests/inference``. The
runnable graph is built with ``onnx.helper`` in the test itself: one ``Conv``
with kernel ``(96, 8)`` and stride ``(1, 8)`` from 3 to ``C`` channels, then
``Reshape`` and ``Transpose`` to ``[1, T, C]``, stamped with the full
``metadata_props`` contract. No torch, no published weights.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from nomikos_inference.architectures.ppocr_rec.adapter import (
    PPOCRRecUnavailableError,
    TranscribeLineFailure,
    _decode_greedy,
    _load_session,
    _response_from_decoded,
    run_ppocr_rec_transcribe_many,
)
from nomikos_inference.contracts.common import InferenceTask, LineCrop, RegistryArchitecture
from nomikos_inference.contracts.transcribe import TranscribeRunResponse
from nomikos_inference.jobs.runner import run_model

CHARSET = ["", "a", "b", "c"]
CLASSES = len(CHARSET)


def _png_line(width: int = 40, height: int = 12) -> bytes:
    output = BytesIO()
    Image.new("L", (width, height), 255).save(output, format="PNG")
    return output.getvalue()


def _write_graph(
    path: Path,
    *,
    charset: list[str] = CHARSET,
    temperature: str = "1.0",
    format: str = "ppocr-rec-onnx-v1",
    blank_index: str = "0",
    extra_props: dict[str, str] | None = None,
) -> Path:
    """Build the tiny contract-conformant graph: Conv, Squeeze, Transpose."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    classes = len(charset)
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 96, "width"])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "time", classes])
    weight = helper.make_tensor(
        "conv_weight", TensorProto.FLOAT, [classes, 3, 96, 8], np.zeros(classes * 3 * 96 * 8)
    )
    # Increasing bias: every frame decodes to the last class, deterministically.
    bias = helper.make_tensor(
        "conv_bias", TensorProto.FLOAT, [classes], np.arange(classes, dtype=np.float32)
    )
    conv = helper.make_node(
        "Conv",
        ["image", "conv_weight", "conv_bias"],
        ["conv_out"],
        kernel_shape=[96, 8],
        strides=[1, 8],
    )
    # Reshape [1, C, 1, T] to [1, C, T] with the time dim read off the Conv
    # output, so the graph stays dynamic in width. Slice and Unsqueeze take
    # their axes as inputs from opset 13 on.
    shape = helper.make_node("Shape", ["conv_out"], ["shape"])
    head = helper.make_node(
        "Slice", ["shape", "slice_starts", "slice_ends", "slice_axes"], ["head"]
    )
    tail = helper.make_node("Gather", ["shape", "gather_index"], ["tail_scalar"], axis=0)
    tail_1d = helper.make_node("Unsqueeze", ["tail_scalar", "unsqueeze_axes"], ["tail"])
    new_shape = helper.make_node("Concat", ["head", "tail"], ["new_shape"], axis=0)
    reshape = helper.make_node("Reshape", ["conv_out", "new_shape"], ["reshaped"])
    transpose = helper.make_node("Transpose", ["reshaped"], ["logits"], perm=[0, 2, 1])
    graph = helper.make_graph(
        [conv, shape, head, tail, tail_1d, new_shape, reshape, transpose],
        "ppocr-rec-tiny",
        [image],
        [logits],
        initializer=[
            weight,
            bias,
            helper.make_tensor("slice_starts", TensorProto.INT64, [1], [0]),
            helper.make_tensor("slice_ends", TensorProto.INT64, [1], [2]),
            helper.make_tensor("slice_axes", TensorProto.INT64, [1], [0]),
            helper.make_tensor("gather_index", TensorProto.INT64, [], [3]),
            helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    import json

    props = {
        "format": format,
        "architecture": "ppocr_rec",
        "variant": "tiny",
        "input_layout": "NCHW",
        "input_name": "image",
        "output_names": json.dumps(["logits"]),
        "input_channels": "3",
        "line_height": "96",
        "subsampling": "8",
        "time_formula": "T = (((W + 1) // 2 + 1) // 2) // 2",
        "classes": str(classes),
        "blank_index": blank_index,
        "charset": json.dumps(charset),
        "pad": "16",
        "pad_fill": "255",
        "temperature": temperature,
        "seg_type": "baselines",
    }
    props.update(extra_props or {})
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))
    return path


@pytest.fixture
def tiny_graph(tmp_path: Path) -> Path:
    return _write_graph(tmp_path / "model.onnx")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# --- Metadata validation ------------------------------------------------------


def test_load_session_rejects_a_file_that_is_not_an_onnx_graph(tmp_path: Path) -> None:
    artifact = tmp_path / "corrupt.onnx"
    artifact.write_bytes(b"not a protobuf")

    with pytest.raises(
        PPOCRRecUnavailableError, match="unable to load PP-OCR recognition ONNX artifact"
    ):
        _load_session(str(artifact))


def test_wrong_format_is_rejected(tmp_path: Path) -> None:
    artifact = _write_graph(tmp_path / "model.onnx", format="calamari-onnx-v1")

    with pytest.raises(PPOCRRecUnavailableError, match="unsupported.*format"):
        run_ppocr_rec_transcribe_many([_png_line()], checkpoint_path=artifact)


def test_wrong_blank_index_is_rejected(tmp_path: Path) -> None:
    artifact = _write_graph(tmp_path / "model.onnx", blank_index="1")

    with pytest.raises(PPOCRRecUnavailableError, match="unsupported blank index"):
        run_ppocr_rec_transcribe_many([_png_line()], checkpoint_path=artifact)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": "0"},
        {"temperature": "nan"},
        {"temperature": "hot"},
        {"temperature": "-1.0"},
        {"charset": ["a", "b", "c"]},
    ],
)
def test_corrupt_metadata_is_rejected(tmp_path: Path, kwargs: dict) -> None:
    artifact = _write_graph(tmp_path / "model.onnx", **kwargs)  # type: ignore[arg-type]

    with pytest.raises(PPOCRRecUnavailableError, match="metadata"):
        run_ppocr_rec_transcribe_many([_png_line()], checkpoint_path=artifact)


# --- Decode --------------------------------------------------------------------


def test_decode_greedy_drops_blanks_and_collapses_repeats() -> None:
    blank = np.array([0.9, 0.05, 0.03, 0.02], dtype=np.float32)
    bright_a = np.array([0.05, 0.8, 0.1, 0.05], dtype=np.float32)
    dimmer_a = np.array([0.1, 0.6, 0.2, 0.1], dtype=np.float32)
    bright_c = np.array([0.05, 0.1, 0.05, 0.8], dtype=np.float32)
    softmax = np.stack([bright_a, blank, bright_c, bright_c.copy(), dimmer_a])

    text, confidences = _decode_greedy(softmax, charset=CHARSET)

    # The repeated ``c`` collapses into one emission keeping the max frame
    # confidence; the ``a`` after the blank is a new emission.
    # Display order here is Latin, so no bidi movement.
    assert text == "aca"
    assert confidences == pytest.approx([0.8, 0.8, 0.6])


def test_decode_greedy_expands_multi_codepoint_graphemes() -> None:
    bright = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    text, confidences = _decode_greedy(np.stack([bright]), charset=["", "ab", "c"])
    assert text == "ab"
    assert confidences == pytest.approx([0.8, 0.8])


def test_response_from_decoded_aligns_character_confidences() -> None:
    response = _response_from_decoded("ab", [0.5, 0.6])
    assert response.text == "ab"
    assert response.confidence == pytest.approx(0.55)
    assert [entry.char for entry in response.character_confidences] == ["a", "b"]


def test_empty_batch_is_a_client_error_even_when_the_weights_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one line image") as caught:
        run_ppocr_rec_transcribe_many([], checkpoint_path=tmp_path / "absent.onnx")

    assert not isinstance(caught.value, OSError)


# --- End to end on the tiny graph ----------------------------------------------


def test_tiny_graph_transcribes_known_text_with_known_confidence(
    tiny_graph: Path,
) -> None:
    results = run_ppocr_rec_transcribe_many([_png_line()], checkpoint_path=tiny_graph)

    assert len(results) == 1
    response = results[0]
    assert isinstance(response, TranscribeRunResponse)
    # Every frame decodes to the last class, collapsed to one emission.
    assert response.text == "c"
    expected = float(_softmax(np.arange(CLASSES, dtype=np.float64))[-1])
    assert response.confidence == pytest.approx(expected)
    assert [entry.char for entry in response.character_confidences] == ["c"]
    assert response.character_confidences[0].confidence == pytest.approx(expected)


def test_temperature_is_applied_before_the_softmax(tmp_path: Path) -> None:
    artifact = _write_graph(tmp_path / "model.onnx", temperature="2.0")
    results = run_ppocr_rec_transcribe_many([_png_line()], checkpoint_path=artifact)

    response = results[0]
    assert isinstance(response, TranscribeRunResponse)
    assert response.text == "c"
    expected = float(_softmax(np.arange(CLASSES, dtype=np.float64) / 2.0)[-1])
    assert response.confidence == pytest.approx(expected)


# --- Per-line isolation ----------------------------------------------------------


def test_one_undecodable_crop_does_not_take_the_page_down(tiny_graph: Path) -> None:
    results = run_ppocr_rec_transcribe_many(
        [_png_line(), b"not an image", _png_line()],
        checkpoint_path=tiny_graph,
    )

    assert len(results) == 3
    assert isinstance(results[1], TranscribeLineFailure)
    assert results[1].index == 1
    survivors = [result for result in results if isinstance(result, TranscribeRunResponse)]
    assert len(survivors) == 2
    assert all(survivor.text == "c" for survivor in survivors)


def test_batch_where_every_line_failed_reraises_the_original_error(
    tiny_graph: Path,
) -> None:
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        run_ppocr_rec_transcribe_many(
            [b"not an image", b"also not an image"],
            checkpoint_path=tiny_graph,
        )


# --- Runner dispatch -------------------------------------------------------------


@pytest.fixture
def ppocr_runner(monkeypatch: pytest.MonkeyPatch):
    """Wire ``run_model`` to a PP-OCR recognition entry without touching weights."""
    monkeypatch.setattr("nomikos_inference.jobs.runner.validate_image_bytes", lambda *_args: None)
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.validate_request_params", lambda *_args: None
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.get_inference_settings",
        lambda: SimpleNamespace(inference_registry_path=Path("registry.yaml")),
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_registry_entry",
        lambda **_kwargs: SimpleNamespace(
            architecture=RegistryArchitecture.ppocr_rec,
            line_crop=LineCrop.polygon_white,
            line_crop_padding=12,
            versions={
                "stable": SimpleNamespace(
                    weights_source="file://unused",
                    hub_revision=None,
                    artifact_sha256=None,
                )
            },
        ),
    )
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_weights_source",
        lambda *_args, **_kwargs: Path("unused.onnx"),
    )

    def run(params: dict | None, image_bytes: bytes | None = None):
        return run_model(
            task=InferenceTask.transcribe,
            registry_model_id="model",
            registry_tag="stable",
            image_bytes=image_bytes if image_bytes is not None else _png_line(),
            params=params,
        )

    return run


def _line_params(count: int) -> dict:
    return {
        "lines": [
            {
                "line_id": f"line-{index}",
                "line_index": index,
                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
            }
            for index in range(count)
        ]
    }


def test_runner_dispatches_single_and_batched_lines_to_ppocr_rec(
    ppocr_runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nomikos_inference.jobs import runner

    seen: list[tuple[str, int]] = []

    def fake_many(line_images: list[bytes], **_kwargs):
        seen.append(("many", len(line_images)))
        return [_response_from_decoded("c", [0.9]) for _ in line_images]

    def fake_single(image_bytes: bytes, **_kwargs):
        seen.append(("single", 1))
        return _response_from_decoded("c", [0.9])

    monkeypatch.setattr(runner, "run_ppocr_rec_transcribe_many", fake_many)
    monkeypatch.setattr(runner, "run_ppocr_rec_transcribe", fake_single)

    single = ppocr_runner(None)
    assert isinstance(single, TranscribeRunResponse)
    assert single.text == "c"

    batch = ppocr_runner(_line_params(2))
    assert [line.output.text for line in batch.lines if line.output] == ["c", "c"]
    assert ("many", 2) in seen
    assert ("single", 1) in seen


def test_runner_rejects_an_unsupported_transcribe_architecture(
    ppocr_runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nomikos_inference.jobs.runner.resolve_registry_entry",
        lambda **_kwargs: SimpleNamespace(
            architecture=RegistryArchitecture.blla,
            line_crop=None,
            line_crop_padding=None,
            versions={
                "stable": SimpleNamespace(
                    weights_source="file://unused",
                    hub_revision=None,
                    artifact_sha256=None,
                )
            },
        ),
    )

    with pytest.raises(ValueError, match="unsupported transcribe architecture"):
        ppocr_runner(None)
