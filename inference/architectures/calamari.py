"""Calamari OCR inference adapter."""

from __future__ import annotations

import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inference.architectures import REPO_ROOT
from inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse


class CalamariUnavailableError(RuntimeError):
    """Raised when the optional Calamari runtime is not installed."""


SUPPORT_CALAMARI_ROOT = REPO_ROOT / "_support_repo" / "calamari"
SUPPORT_CALAMARI_VERSION = "2.3.1-local"


def _ensure_support_calamari_importable() -> None:
    if not SUPPORT_CALAMARI_ROOT.is_dir():
        raise CalamariUnavailableError(
            f"local Calamari architecture source not found: {SUPPORT_CALAMARI_ROOT}"
        )

    version_path = SUPPORT_CALAMARI_ROOT / "calamari_ocr" / "version.py"
    if not version_path.exists():
        version_path.write_text(f'__version__ = "{SUPPORT_CALAMARI_VERSION}"\n', encoding="utf-8")

    if str(SUPPORT_CALAMARI_ROOT) not in sys.path:
        sys.path.insert(0, str(SUPPORT_CALAMARI_ROOT))


@lru_cache(maxsize=4)
def _load_predictor(checkpoint_path: str) -> Any:
    _ensure_support_calamari_importable()
    try:
        from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
    except ImportError as exc:
        raise CalamariUnavailableError(
            "local Calamari architecture dependencies are required for real inference; "
            "install the project with the `calamari` extra"
        ) from exc

    params = PredictorParams(silent=True, progress_bar=False)
    params.pipeline.num_processes = 1
    return Predictor.from_checkpoint(params, checkpoint_path)


def _line_image_array(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("L"))


def _prediction_result(sample: Any) -> Any:
    """Unwrap tfaip Sample objects returned by predict_raw."""
    outputs = getattr(sample, "outputs", None)
    if outputs is not None and hasattr(outputs, "sentence"):
        return outputs
    return sample


def _characters_with_confidence(text: str, prediction: Any) -> list[CharacterConfidence]:
    positions = getattr(prediction, "positions", None) or []
    confidences: list[float] = []
    for position in positions:
        chars = getattr(position, "chars", None) or []
        if chars:
            confidences.append(float(getattr(chars[0], "probability", 1.0)))

    if len(confidences) != len(text):
        avg = float(getattr(prediction, "avg_char_probability", 1.0) or 1.0)
        confidences = [avg for _ in text]

    return [
        CharacterConfidence(char=char, confidence=max(0.0, min(1.0, confidence)))
        for char, confidence in zip(text, confidences, strict=True)
    ]


def _response_from_prediction(prediction: Any) -> TranscribeRunResponse:
    result = _prediction_result(prediction)
    text = str(getattr(result, "sentence", "") or "")
    confidence = float(getattr(result, "avg_char_probability", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    return TranscribeRunResponse(
        text=text,
        confidence=confidence,
        character_confidences=_characters_with_confidence(text, result),
    )


def run_calamari_transcribe_many(
    line_images: list[bytes],
    *,
    checkpoint_path: Path,
) -> list[TranscribeRunResponse]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Calamari checkpoint not found: {checkpoint_path}")
    if not line_images:
        raise ValueError("at least one line image is required")

    predictor = _load_predictor(str(checkpoint_path))
    predictions = predictor.predict_raw([_line_image_array(image) for image in line_images])
    return [_response_from_prediction(prediction) for prediction in predictions]


def run_calamari_transcribe(
    image_bytes: bytes,
    *,
    checkpoint_path: Path,
) -> TranscribeRunResponse:
    return run_calamari_transcribe_many(
        [image_bytes],
        checkpoint_path=checkpoint_path,
    )[0]
