"""Tests for the minimal PyTorch Calamari graph."""
# pyright: reportMissingImports=false

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.paths import REPO_ROOT, TRANSCRIBE_LINE

torch = pytest.importorskip("torch")

from src.model.inference_export.calamari import (  # noqa: E402
    CalamariTorchConfig,
    CalamariTorchLayerConfig,
    CalamariTorchModel,
)
from inference.architectures.calamari import (  # noqa: E402
    preprocess_line_image_to_calamari_tensor,
    run_calamari_transcribe,
)
from inference.architectures.calamari.preprocessing import (  # noqa: E402
    preprocess_line_image_bytes_to_calamari_tensor,
)


def test_converted_checkpoint_carries_runtime_metadata() -> None:
    checkpoint = torch.load(
        REPO_ROOT / "src/hf/local/syriac/calamari/v1/stable/best.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["format"] == "calamari-pytorch-v1"
    assert checkpoint["classes"] == 47
    assert checkpoint["line_height"] == 48
    assert len(checkpoint["charset"]) == 47


def test_forward_returns_calamari_logit_shapes_and_class_roll() -> None:
    torch.manual_seed(13)
    model = CalamariTorchModel(_tiny_config())
    model.eval()
    image = _transcribe_fixture_tensor()
    image_length = torch.tensor([image.shape[1]])

    outputs = model(image, image_lengths=image_length)
    expected_time = _tiny_config().downscaled_sequence_lengths(image_length).item()

    assert outputs["blank_last_logits"].shape == (1, expected_time, 6)
    assert outputs["logits"].shape == (1, expected_time, 6)
    assert outputs["softmax"].shape == (1, expected_time, 6)
    assert outputs["out_len"].tolist() == [expected_time]
    assert torch.allclose(
        outputs["logits"],
        torch.roll(outputs["blank_last_logits"], shifts=1, dims=-1),
    )


def test_preprocess_line_image_matches_vendored_calamari_processors() -> None:
    actual = preprocess_line_image_to_calamari_tensor(TRANSCRIBE_LINE)

    assert actual.shape == (1, 291, 48, 1)
    assert actual.dtype == np.uint8
    assert actual.min() >= 0
    assert actual.max() <= 255


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


def _tiny_config() -> CalamariTorchConfig:
    return CalamariTorchConfig(
        layers=(
            CalamariTorchLayerConfig(
                kind="conv2d",
                name="conv2d_0",
                filters=2,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            CalamariTorchLayerConfig(
                kind="maxpool2d",
                name="maxpool2d_0",
                pool_size=(2, 2),
                strides=(-1, -1),
                padding="same",
            ),
            CalamariTorchLayerConfig(
                kind="bilstm",
                name="lstm_0",
                hidden_nodes=3,
                merge_mode="concat",
            ),
            CalamariTorchLayerConfig(
                kind="dropout",
                name="dropout_0",
                rate=0.5,
            ),
        ),
        classes=6,
    )


def _transcribe_fixture_tensor() -> torch.Tensor:
    tensor = preprocess_line_image_to_calamari_tensor(TRANSCRIBE_LINE)
    return torch.from_numpy(tensor.astype(np.float32))
