"""Focused Calamari ONNX export and runtime parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.paths import REPO_ROOT, TRANSCRIBE_LINE

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from inference.architectures.calamari.adapter import run_calamari_transcribe  # noqa: E402
from src.model.inference_export.calamari import (  # noqa: E402
    export_calamari_onnx,
    load_calamari_checkpoint,
)


CHECKPOINT = REPO_ROOT / "src/hf/local/syriac/calamari/v1/stable/best.pt"


def test_export_embeds_codec_metadata_and_handles_odd_even_widths(tmp_path: Path) -> None:
    destination = tmp_path / "calamari.onnx"
    metadata = export_calamari_onnx(CHECKPOINT, destination)
    model = onnx.load(destination)
    embedded = {entry.key: entry.value for entry in model.metadata_props}

    assert metadata.classes == 47
    assert embedded["format"] == "calamari-onnx-v1"
    assert embedded["line_height"] == "48"
    assert embedded["blank_index"] == "0"
    assert len(json.loads(embedded["charset"])) == 47

    reference, _ = load_calamari_checkpoint(CHECKPOINT)
    reference.eval()
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    for width in (7, 8, 17, 18):
        image = np.random.default_rng(width).random((1, width, 48, 1), dtype=np.float32) * 255
        lengths = np.asarray([width], dtype=np.int64)
        with torch.no_grad():
            expected = reference(
                torch.from_numpy(image),
                image_lengths=torch.from_numpy(lengths),
            )
        actual_logits, actual_lengths = session.run(
            ["logits", "out_len"],
            {"image": image, "image_lengths": lengths},
        )
        np.testing.assert_allclose(
            actual_logits,
            expected["logits"].numpy(),
            rtol=1e-4,
            atol=2e-4,
        )
        np.testing.assert_array_equal(actual_lengths, expected["out_len"].numpy())


def test_onnx_adapter_preserves_legacy_decoding(tmp_path: Path) -> None:
    destination = tmp_path / "calamari.onnx"
    export_calamari_onnx(CHECKPOINT, destination)

    response = run_calamari_transcribe(
        TRANSCRIBE_LINE.read_bytes(),
        checkpoint_path=destination,
    )

    assert isinstance(response.text, str)
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.character_confidences) == len(response.text)
