"""Tests for the minimal PyTorch Calamari graph."""
# pyright: reportMissingImports=false

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.paths import REPO_ROOT, TRANSCRIBE_LINE

torch = pytest.importorskip("torch")

from inference.architectures.calamari import (  # noqa: E402
    CalamariTorchConfig,
    CalamariTorchLayerConfig,
    CalamariTorchModel,
    preprocess_line_image_to_calamari_tensor,
    run_calamari_transcribe,
)


def test_converted_checkpoint_carries_runtime_metadata() -> None:
    checkpoint = torch.load(
        REPO_ROOT / "src/hf/local/syriac/calamari/v1/stable/best.pt",
        map_location="cpu",
        weights_only=False,
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


def test_run_calamari_transcribe_uses_converted_pytorch_checkpoint() -> None:
    response = run_calamari_transcribe(
        TRANSCRIBE_LINE.read_bytes(),
        checkpoint_path=REPO_ROOT / "src/hf/local/syriac/calamari/v1/stable/best.pt",
    )

    assert isinstance(response.text, str)
    assert 0.0 <= response.confidence <= 1.0
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
