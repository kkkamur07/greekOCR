"""Torch-graph-vs-published-artifact parity for Calamari."""
# pyright: reportMissingImports=false

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.paths import REPO_ROOT, TRANSCRIBE_LINE

torch = pytest.importorskip("torch")

from inference.architectures.calamari import run_calamari_transcribe  # noqa: E402
from inference.architectures.calamari.preprocessing import (  # noqa: E402
    preprocess_line_image_bytes_to_calamari_tensor,
)


def test_the_published_graph_and_artifact_decode_the_same_line() -> None:
    """Parity, on the two files that were published together.

    The Torch graph is the oracle and the ``.onnx`` is what ships, so this is
    the claim ADR 0006 rests on: converting the graph did not change what a
    researcher reads. Text must match exactly - a CTC decode that agrees only
    approximately has already changed a character somewhere.
    """
    from src.model.inference_export.calamari import load_calamari_checkpoint
    from inference.architectures.calamari.adapter import _decode_greedy

    checkpoint = REPO_ROOT / "src/hf/cache/syriac-calamari-v1/stable/best.pt"
    artifact = REPO_ROOT / "src/hf/cache/syriac-calamari-v1/stable/best.onnx"
    if not (checkpoint.is_file() and artifact.is_file()):
        pytest.skip("published Calamari artifacts are not cached locally")

    model, metadata = load_calamari_checkpoint(checkpoint)
    image = preprocess_line_image_bytes_to_calamari_tensor(
        TRANSCRIBE_LINE.read_bytes(), line_height=metadata.line_height
    )
    with torch.inference_mode():
        softmax = model(
            torch.from_numpy(image.astype(np.float32)),
            image_lengths=torch.tensor([image.shape[1]]),
        )["softmax"][0].numpy()
    reference_text, _ = _decode_greedy(softmax, charset=list(metadata.charset))

    response = run_calamari_transcribe(TRANSCRIBE_LINE.read_bytes(), checkpoint_path=artifact)

    assert response.text == reference_text
    assert response.text  # a shared empty decode would satisfy the line above
    assert len(response.character_confidences) == len(response.text)
