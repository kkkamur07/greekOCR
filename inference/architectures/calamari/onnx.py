"""Calamari inference through ONNX Runtime."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)
from inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse


class CalamariUnavailableError(RuntimeError):
    """Raised when a Calamari runtime artifact cannot be used."""


def _file_fingerprint(path: Path) -> tuple[int, int]:
    """Cache-key component so replaced artifact files are reloaded."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _load_onnx_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[object, list[str], int]:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("format") != "calamari-onnx-v1":
            raise CalamariUnavailableError("unsupported Calamari ONNX artifact format")
        classes = _metadata_int(metadata, "classes", minimum=2)
        line_height = _metadata_int(metadata, "line_height", minimum=1)
        if metadata.get("blank_index") != "0":
            raise CalamariUnavailableError(
                "Calamari ONNX artifact has an unsupported blank index"
            )
        try:
            temperature = float(metadata["temperature"])
        except (KeyError, TypeError, ValueError) as error:
            raise CalamariUnavailableError(
                "Calamari ONNX artifact has invalid temperature metadata"
            ) from error
        if not math.isfinite(temperature):
            raise CalamariUnavailableError(
                "Calamari ONNX artifact has invalid temperature metadata"
            )
        charset_value = metadata.get("charset")
        if charset_value is None:
            raise CalamariUnavailableError("Calamari ONNX artifact has no codec metadata")
        charset = json.loads(charset_value)
        if (
            not isinstance(charset, list)
            or len(charset) != classes
            or not all(isinstance(character, str) for character in charset)
        ):
            raise CalamariUnavailableError("Calamari ONNX artifact has invalid codec metadata")
        input_names = {input_.name for input_ in session.get_inputs()}
        if not {"image", "image_lengths"}.issubset(input_names):
            raise CalamariUnavailableError("Calamari ONNX artifact has incompatible inputs")
        output_names = {output.name for output in session.get_outputs()}
        if not {"logits", "out_len"}.issubset(output_names):
            raise CalamariUnavailableError("Calamari ONNX artifact has incompatible outputs")
        return session, charset, line_height
    except CalamariUnavailableError:
        raise
    except Exception as error:
        raise CalamariUnavailableError("unable to load Calamari ONNX artifact") from error


def _metadata_int(metadata: Mapping[str, str], key: str, *, minimum: int) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as error:
        raise CalamariUnavailableError(f"Calamari ONNX metadata has invalid {key}") from error
    if value < minimum:
        raise CalamariUnavailableError(f"Calamari ONNX metadata has invalid {key}")
    return value


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

    # Trim edge whitespace together with its confidences so the per-character
    # confidence alignment survives (a bare ``str.strip`` would desync them).
    while text_parts and text_parts[0].isspace():
        text_parts.pop(0)
        confidences.pop(0)
    while text_parts and text_parts[-1].isspace():
        text_parts.pop()
        confidences.pop()
    return "".join(text_parts), confidences


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


def run_calamari_onnx_transcribe_many(
    line_images: list[bytes],
    *,
    checkpoint_path: Path,
) -> list[TranscribeRunResponse]:
    """Run the self-contained ONNX artifact for one or more line images."""

    session, charset, line_height = _load_onnx_session(
        str(checkpoint_path), _file_fingerprint(checkpoint_path)
    )
    responses: list[TranscribeRunResponse] = []
    for image_bytes in line_images:
        image = preprocess_line_image_bytes_to_calamari_tensor(
            image_bytes,
            line_height=line_height,
        ).astype(np.float32, copy=False)
        outputs = session.run(
            ["logits", "out_len"],
            {
                "image": image,
                "image_lengths": np.asarray([image.shape[1]], dtype=np.int64),
            },
        )
        logits = np.asarray(outputs[0], dtype=np.float32)[0]
        logits -= np.max(logits, axis=-1, keepdims=True)
        softmax = np.exp(logits)
        softmax /= np.sum(softmax, axis=-1, keepdims=True)
        text, confidences = _decode_greedy(softmax, charset=charset)
        responses.append(_response_from_decoded(text, confidences))
    return responses


def run_calamari_onnx_transcribe(
    image_bytes: bytes,
    *,
    checkpoint_path: Path,
) -> TranscribeRunResponse:
    return run_calamari_onnx_transcribe_many(
        [image_bytes],
        checkpoint_path=checkpoint_path,
    )[0]


__all__ = [
    "CalamariUnavailableError",
    "run_calamari_onnx_transcribe",
    "run_calamari_onnx_transcribe_many",
]
