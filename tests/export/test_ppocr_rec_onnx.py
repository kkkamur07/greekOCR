"""PP-OCRv6 recognition ONNX export and runtime parity tests.

They live under ``tests/export`` rather than ``tests/inference`` because they
import Torch and kraken: the kraken graph is the export-time oracle, and
nothing a researcher installs can import it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
kraken_models = pytest.importorskip("kraken.models")

from nomikos_inference.export.ppocr_rec import (  # noqa: E402
    export_ppocr_rec_onnx,
)

# Widths where the backbone yields one frame more than W // 8 (SAME-padded
# strided convolutions); the contract pins ONNX == Torch, not the division.
PLUS_ONE_WIDTHS = (15, 31)


def _tiny_checkpoint(path: Path) -> None:
    """Save a randomly initialised tiny PP-OCRv6 model with a fixed seed."""
    from kraken.lib.ppocr.model import PPOCRv6Model
    from kraken.models.writers import write_safetensors

    torch.manual_seed(7)
    codec = {chr(0x0627 + index): [index + 1] for index in range(11)}
    model = PPOCRv6Model(
        variant="tiny",
        num_classes=12,
        height=96,
        codec=codec,
        seg_type="baselines",
    )
    model.eval()
    write_safetensors([model], path)


def _metadata_map(path: Path) -> dict[str, str]:
    return {entry.key: entry.value for entry in onnx.load(path).metadata_props}


def test_export_metadata_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.safetensors"
    destination = tmp_path / "model.onnx"
    _tiny_checkpoint(checkpoint)

    report = export_ppocr_rec_onnx(checkpoint, destination)

    assert report.classes == 12
    assert report.variant == "tiny"
    assert report.line_height == 96
    embedded = _metadata_map(destination)
    assert embedded["format"] == "ppocr-rec-onnx-v1"
    assert embedded["architecture"] == "ppocr_rec"
    assert embedded["variant"] == "tiny"
    assert embedded["input_layout"] == "NCHW"
    assert embedded["input_name"] == "image"
    assert json.loads(embedded["output_names"]) == ["logits"]
    assert embedded["input_channels"] == "3"
    assert embedded["line_height"] == "96"
    assert embedded["subsampling"] == "8"
    assert embedded["time_formula"] == "T = (((W + 1) // 2 + 1) // 2) // 2"
    assert embedded["classes"] == "12"
    assert embedded["blank_index"] == "0"
    assert embedded["pad"] == "16"
    assert embedded["pad_fill"] == "255"
    assert embedded["temperature"] == "1.0"
    assert embedded["seg_type"] == "baselines"
    assert embedded["opset_version"] == "17"
    assert embedded["source_format"] == "kraken-safetensors"
    assert len(embedded["source_sha256"]) == 64
    assert embedded["kraken_version"]
    assert embedded["torch_version"]


def test_export_parity_and_dynamic_width(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.safetensors"
    destination = tmp_path / "model.onnx"
    _tiny_checkpoint(checkpoint)
    export_ppocr_rec_onnx(checkpoint, destination)

    models = kraken_models.load_models(str(checkpoint))
    reference = models[0]
    reference.eval()
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    for width in (32, 100, 321, *PLUS_ONE_WIDTHS):
        image = np.random.default_rng(width).random((1, 3, 96, width), dtype=np.float32)
        with torch.no_grad():
            expected, _ = reference(torch.from_numpy(image))
        actual = session.run(["logits"], {"image": image})[0]
        assert list(actual.shape) == [1, int(expected.shape[3]), 12]
        np.testing.assert_allclose(
            actual[0],
            expected.numpy()[0, :, 0, :].transpose(1, 0),
            rtol=1e-4,
            atol=2e-4,
        )


def test_export_charset_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.safetensors"
    destination = tmp_path / "model.onnx"
    _tiny_checkpoint(checkpoint)
    export_ppocr_rec_onnx(checkpoint, destination)

    charset = json.loads(_metadata_map(destination)["charset"])
    assert len(charset) == 12
    assert charset[0] == ""
    models = kraken_models.load_models(str(checkpoint))
    c2l = models[0].codec.c2l
    assert {grapheme: [index] for index, grapheme in enumerate(charset) if index} == c2l


def test_export_real_checkpoint(tmp_path: Path) -> None:
    checkpoint = os.environ.get("PPOCR_REC_CHECKPOINT")
    if not checkpoint:
        pytest.skip("PPOCR_REC_CHECKPOINT is not set")
    source = Path(checkpoint)
    if not source.is_file():
        pytest.skip(f"PPOCR_REC_CHECKPOINT has no file: {source}")

    report = export_ppocr_rec_onnx(source, tmp_path / "model.onnx")
    assert report.classes == 1623
    assert report.line_height == 96
