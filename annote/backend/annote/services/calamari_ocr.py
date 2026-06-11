"""Calamari OCR prediction for pairing assist."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from annote.schemas.annotation import ModelCharacterConfidence, PageAnnotation, Segment
from annote.services.annotation_store import load_annotation, save_annotation
from annote.services.page_image import load_working_page_rgb, resolve_source_page_image
from annote.services.processing.rectify import rectify
from annote.settings import get_settings

_predictor = None


@dataclass(frozen=True)
class SegmentOcrResult:
    text: str
    char_confidence: list[ModelCharacterConfidence]


def _calamari_install_hint(cause: ImportError) -> str:
    return (
        f"Calamari is not available in this Python ({sys.executable}): {cause}. "
        "Stop the server, then from annote/backend run: "
        "source .venv/bin/activate && pip install -e '.[calamari]' && annote"
    )


def _checkpoint_path() -> Path:
    return get_settings().calamari.checkpoint


def _checkpoint_model_path(checkpoint: Path) -> Path:
    path = str(checkpoint)
    return Path(path.removesuffix(".json"))


def _checkpoint_metadata_path(checkpoint: Path) -> Path:
    model_path = _checkpoint_model_path(checkpoint)
    return Path(f"{model_path}.json")


def _checkpoint_exists(checkpoint: Path) -> bool:
    """True when Calamari model weights and companion .json metadata both exist."""
    model_path = _checkpoint_model_path(checkpoint)
    metadata_path = _checkpoint_metadata_path(checkpoint)
    if not metadata_path.is_file():
        return False
    if model_path.is_file():
        return True
    if model_path.is_dir():
        return (model_path / "saved_model.pb").is_file() or (model_path / "keras_metadata.pb").is_file()
    return False


def _force_serial_pipeline(pipeline_params) -> None:
    for name, value in (
        ("num_processes", 1),
        ("batch_size", 1),
        ("run_parallel", False),
        ("num_threads", 1),
    ):
        if hasattr(pipeline_params, name):
            setattr(pipeline_params, name, value)


def _disable_parallel_preprocessing(predictor) -> None:
    """Run image/text processors in-process (required for uvicorn on macOS)."""
    data_params = predictor.data.params
    for name in ("pre_proc", "post_proc"):
        pipeline_params = getattr(data_params, name, None)
        if pipeline_params is not None:
            _force_serial_pipeline(pipeline_params)


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    checkpoint = _checkpoint_model_path(_checkpoint_path())
    metadata_path = _checkpoint_metadata_path(checkpoint)
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Calamari checkpoint metadata not found: {metadata_path}. "
            "Copy best.ckpt.json from your calamari-train output next to the model directory."
        )
    if not _checkpoint_exists(checkpoint):
        raise RuntimeError(f"Calamari checkpoint not found: {checkpoint}")

    try:
        from calamari_ocr.ocr.predict.params import PredictorParams
        from calamari_ocr.ocr.predict.predictor import Predictor
    except ImportError as e:
        raise RuntimeError(_calamari_install_hint(e)) from e

    try:
        params = PredictorParams(silent=True)
        # Avoid multiprocessing spawn issues inside uvicorn workers.
        _force_serial_pipeline(params.pipeline)
        _predictor = Predictor.from_checkpoint(params, checkpoint=str(checkpoint))
        _disable_parallel_preprocessing(_predictor)
    except Exception as e:
        raise RuntimeError(f"Failed to load Calamari checkpoint {checkpoint}: {e}") from e
    return _predictor


def _prediction_outputs(sample):
    return getattr(sample, "outputs", sample)


def _extract_char_confidence(outputs, codec) -> list[ModelCharacterConfidence]:
    labels = getattr(outputs, "labels", None)
    positions = getattr(outputs, "positions", None)
    if not labels or not positions:
        return []

    decoded = codec.decode(labels)
    confidences: list[ModelCharacterConfidence] = []
    for index, position in enumerate(positions):
        label = labels[index]
        chars = getattr(position, "chars", None) or []
        match = next((char for char in chars if char.label == label), chars[0] if chars else None)
        probability = float(match.probability) if match is not None else 0.0
        char = decoded[index] if index < len(decoded) else ""
        confidences.append(ModelCharacterConfidence(char=char, probability=probability))
    return confidences


def _extract_prediction(sample, codec) -> SegmentOcrResult:
    outputs = _prediction_outputs(sample)
    sentence = str(getattr(outputs, "sentence", "") or "")
    char_confidence = _extract_char_confidence(outputs, codec)
    if not sentence and char_confidence:
        sentence = "".join(item.char for item in char_confidence)
    return SegmentOcrResult(text=sentence, char_confidence=char_confidence)


def _rectified_grayscale(page_image: np.ndarray, segment: Segment) -> np.ndarray:
    crop = rectify(page_image, segment.model_dump())
    if crop.ndim == 3:
        pil = Image.fromarray(crop).convert("L")
        return np.array(pil, dtype=np.uint8)
    return crop.astype(np.uint8)


def predict_segment(page_image: np.ndarray, segment: Segment) -> SegmentOcrResult:
    """Run OCR on a rectified grayscale crop for one segment."""
    gray = _rectified_grayscale(page_image, segment)
    predictor = _get_predictor()
    predictions = list(predictor.predict_raw([gray]))
    if not predictions:
        return SegmentOcrResult(text="", char_confidence=[])
    codec = predictor.data.params.codec
    return _extract_prediction(predictions[0], codec)


def predict_segment_text(page_image: np.ndarray, segment: Segment) -> str:
    return predict_segment(page_image, segment).text


def _segment_ocr_update(result: SegmentOcrResult, timestamp: str) -> dict:
    return {
        "model_transcription": result.text,
        "model_transcription_confidence": result.char_confidence or None,
        "model_transcription_at": timestamp,
    }


def run_segment_ocr(data_root: Path, stem: str, segment_id: str) -> PageAnnotation:
    """Predict text for one segment and persist model fields on the annotation."""
    annotation = load_annotation(data_root, stem)
    segment = next((s for s in annotation.segments if s.id == segment_id), None)
    if segment is None:
        raise LookupError(f"Segment not found: {segment_id}")

    if resolve_source_page_image(data_root, stem) is None:
        raise FileNotFoundError(f"Page image not found: {stem}")

    page_image, _ = load_working_page_rgb(data_root, stem)
    result = predict_segment(page_image, segment)
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    segments = []
    for seg in annotation.segments:
        if seg.id == segment_id:
            seg = seg.model_copy(update=_segment_ocr_update(result, now))
        segments.append(seg)

    updated = annotation.model_copy(update={"segments": segments})
    return save_annotation(data_root, stem, updated)


def ocr_page_events(data_root: Path, stem: str):
    """Yield progress events while running OCR on every segment on the page."""
    from annote.schemas.ocr import OcrDoneEvent, OcrProgressEvent, OcrResult

    annotation = load_annotation(data_root, stem)
    segments = list(annotation.segments)
    total = len(segments)

    if total == 0:
        yield OcrDoneEvent(result=OcrResult(processed_count=0))
        return

    if resolve_source_page_image(data_root, stem) is None:
        raise FileNotFoundError(f"Page image not found: {stem}")

    page_image, _ = load_working_page_rgb(data_root, stem)
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    updated_segments: list[Segment] = []

    for index, segment in enumerate(segments, start=1):
        result = predict_segment(page_image, segment)
        updated_segments.append(segment.model_copy(update=_segment_ocr_update(result, now)))
        save_annotation(data_root, stem, annotation.model_copy(update={"segments": updated_segments + segments[index:]}))
        yield OcrProgressEvent(
            current=index,
            total=total,
            segment_number=segment.number,
            segment_id=segment.id,
        )

    save_annotation(data_root, stem, annotation.model_copy(update={"segments": updated_segments}))
    yield OcrDoneEvent(result=OcrResult(processed_count=total))
