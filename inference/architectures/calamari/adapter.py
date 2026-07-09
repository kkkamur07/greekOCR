"""Calamari OCR inference adapter backed by the local PyTorch graph."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from inference.architectures.calamari.config import CalamariTorchConfig, CalamariTorchLayerConfig
from inference.architectures.calamari.model import CalamariTorchModel
from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)
from inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse


class CalamariUnavailableError(RuntimeError):
    """Raised when a Calamari PyTorch checkpoint cannot be used."""


def _default_config(*, classes: int) -> CalamariTorchConfig:
    return CalamariTorchConfig(
        layers=(
            CalamariTorchLayerConfig(
                kind="conv2d",
                name="conv2d_0",
                filters=40,
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
                kind="conv2d",
                name="conv2d_1",
                filters=60,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            CalamariTorchLayerConfig(
                kind="maxpool2d",
                name="maxpool2d_1",
                pool_size=(2, 2),
                strides=(-1, -1),
                padding="same",
            ),
            CalamariTorchLayerConfig(
                kind="bilstm",
                name="lstm_0",
                hidden_nodes=200,
                merge_mode="concat",
            ),
            CalamariTorchLayerConfig(
                kind="dropout",
                name="dropout_0",
                rate=0.5,
            ),
        ),
        classes=classes,
    )


@lru_cache(maxsize=4)
def _load_checkpoint(checkpoint_path: str) -> tuple[CalamariTorchModel, list[str] | None, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "calamari-pytorch-v1":
        raise CalamariUnavailableError(f"unsupported Calamari checkpoint format: {checkpoint_path}")

    classes = int(checkpoint["classes"])
    line_height = int(checkpoint.get("line_height", 48))
    model = CalamariTorchModel(_default_config(classes=classes))
    model.eval()
    dummy = torch.zeros((1, 4, line_height, 1), dtype=torch.float32)
    _ = model(dummy, image_lengths=torch.tensor([4]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("charset"), line_height


def _decode_greedy(
    softmax: np.ndarray,
    *,
    charset: list[str],
) -> tuple[str, list[float]]:
    labels = np.argmax(softmax, axis=1)
    text_parts: list[str] = []
    confidences: list[float] = []
    last_label = 0

    for index, label in enumerate(labels):
        label = int(label)
        if label == 0:
            last_label = label
            continue
        if label != last_label:
            char = charset[label] if label < len(charset) else ""
            if char:
                text_parts.append(char)
                confidences.append(float(softmax[index, label]))
        elif confidences:
            confidences[-1] = max(confidences[-1], float(softmax[index, label]))
        last_label = label

    return "".join(text_parts).strip(), confidences


def _response_from_decoded(text: str, confidences: list[float]) -> TranscribeRunResponse:
    if len(confidences) != len(text):
        confidences = [float(np.mean(confidences)) if confidences else 0.0 for _ in text]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return TranscribeRunResponse(
        text=text,
        confidence=max(0.0, min(1.0, confidence)),
        character_confidences=[
            CharacterConfidence(char=char, confidence=max(0.0, min(1.0, confidence)))
            for char, confidence in zip(text, confidences, strict=True)
        ],
    )


def run_calamari_transcribe_many(
    line_images: list[bytes],
    *,
    checkpoint_path: Path,
) -> list[TranscribeRunResponse]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Calamari checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix != ".pt":
        raise CalamariUnavailableError(
            f"Calamari runtime now requires a PyTorch .pt checkpoint: {checkpoint_path}"
        )
    if not line_images:
        raise ValueError("at least one line image is required")

    model, charset, line_height = _load_checkpoint(str(checkpoint_path))
    if not charset:
        raise CalamariUnavailableError(
            f"Calamari checkpoint has no codec metadata: {checkpoint_path}"
        )

    responses: list[TranscribeRunResponse] = []
    with torch.inference_mode():
        for image_bytes in line_images:
            image = preprocess_line_image_bytes_to_calamari_tensor(
                image_bytes,
                line_height=line_height,
            )
            image_tensor = torch.from_numpy(image.astype(np.float32))
            image_lengths = torch.tensor([image.shape[1]], dtype=torch.long)
            outputs = model(image_tensor, image_lengths=image_lengths)
            softmax = outputs["softmax"][0].detach().cpu().numpy()
            text, confidences = _decode_greedy(softmax, charset=charset)
            responses.append(_response_from_decoded(text, confidences))
    return responses


def run_calamari_transcribe(
    image_bytes: bytes,
    *,
    checkpoint_path: Path,
) -> TranscribeRunResponse:
    return run_calamari_transcribe_many(
        [image_bytes],
        checkpoint_path=checkpoint_path,
    )[0]
