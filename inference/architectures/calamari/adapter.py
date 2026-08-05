"""Calamari transcription on the PyTorch CPU runtime (ADR 0004)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from inference.architectures.artifact import ArtifactHandle, resolve_artifact
from inference.architectures.calamari.checkpoint import load_calamari_checkpoint
from inference.architectures.calamari.model import CalamariTorchModel
from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)
from inference.architectures.isolation import reraise_if_none_survived
from inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse

# The Calamari **Hub artifact** is the native ``.pt`` checkpoint the training
# run produced. There is no second format to accept since ADR 0004 retired the
# ONNX runtime.
CALAMARI_ARTIFACT_SUFFIXES = frozenset({".pt"})


class CalamariUnavailableError(RuntimeError):
    """Raised when a Calamari runtime artifact cannot be used."""


@dataclass(frozen=True)
class TranscribeLineFailure:
    """One line of a batch that could not be transcribed.

    Returned in place of that line's output instead of raised, so a single
    unusable crop degrades to a per-line error rather than discarding the whole
    page. The original exception rides along because an all-failed batch has to
    re-raise it: the run-error mapping distinguishes a broken artifact (503)
    from a bad request (422), and both would collapse into a generic 500 if the
    cause were flattened to a string here.
    """

    index: int
    error: Exception


@lru_cache(maxsize=4)
def _load_checkpoint(
    checkpoint_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[CalamariTorchModel, list[str], int]:
    """Open a digest-verified checkpoint without unpickling it.

    ``fingerprint`` is part of the cache key rather than an argument the loader
    reads: it is what makes a *replaced* artifact file miss the cache instead of
    serving the previous model for the life of the process.
    """
    try:
        model, metadata = load_calamari_checkpoint(Path(checkpoint_path))
    except ValueError as error:
        message = str(error)
        if "safely load" in message:
            raise CalamariUnavailableError("unable to safely load Calamari checkpoint") from error
        if "state dictionary" in message:
            raise CalamariUnavailableError(
                "Calamari checkpoint state dictionary is incompatible with the runtime"
            ) from error
        raise CalamariUnavailableError(message) from error
    except Exception as error:
        raise CalamariUnavailableError("unable to safely load Calamari checkpoint") from error
    return model, list(metadata.charset), metadata.line_height


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


def _reject_fully_failed_batch(
    results: list[TranscribeRunResponse | TranscribeLineFailure],
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    """Let partial results through, but never an all-failed batch.

    Isolating per-line failures is only safe while at least one line survived.
    If none did, the cause is almost certainly the artifact or the runtime, and
    re-raising the first line's original exception keeps its HTTP mapping (503
    for an unusable runtime, 422 for an unusable request) instead of handing the
    caller a page of identical per-line errors that looks like a successful run.

    The rule itself lives in ``architectures.isolation`` because BLLA has to
    reach the same verdict from a differently shaped loop.
    """
    failures = [result for result in results if isinstance(result, TranscribeLineFailure)]
    reraise_if_none_survived(
        survivors=len(results) - len(failures),
        first_failure=failures[0].error if failures else None,
    )
    return results


def run_calamari_transcribe_many(
    line_images: list[bytes],
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None = None,
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    handle = resolve_artifact(
        checkpoint_path,
        label="Calamari checkpoint",
        allowed_suffixes=CALAMARI_ARTIFACT_SUFFIXES,
        unusable_error=CalamariUnavailableError,
        unusable_message=(f"Calamari runtime requires a native .pt checkpoint: {checkpoint_path}"),
        artifact_sha256=artifact_sha256,
    )
    if not line_images:
        raise ValueError("at least one line image is required")

    return _reject_fully_failed_batch(_run_torch_transcribe_many(line_images, handle=handle))


def _run_torch_transcribe_many(
    line_images: list[bytes],
    *,
    handle: ArtifactHandle,
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    model, charset, line_height = _load_checkpoint(handle.path, handle.fingerprint)
    if not charset:
        raise CalamariUnavailableError(f"Calamari checkpoint has no codec metadata: {handle.path}")

    responses: list[TranscribeRunResponse | TranscribeLineFailure] = []
    # ``load_calamari_checkpoint`` already called ``eval()``; the model is
    # cached across calls, so assert it here rather than trust the cache.
    model.eval()
    with torch.inference_mode():
        for index, image_bytes in enumerate(line_images):
            # Per-line isolation: the caller decides what a failed line means,
            # and one of them must not end the batch.
            try:
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
            except Exception as error:  # noqa: BLE001 - per-line isolation is the point
                responses.append(TranscribeLineFailure(index=index, error=error))
    return responses


def run_calamari_transcribe(
    image_bytes: bytes,
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None = None,
) -> TranscribeRunResponse:
    result = run_calamari_transcribe_many(
        [image_bytes],
        checkpoint_path=checkpoint_path,
        artifact_sha256=artifact_sha256,
    )[0]
    if isinstance(result, TranscribeLineFailure):
        # Unreachable: a one-line batch that failed already re-raised above.
        raise result.error
    return result


__all__ = [
    "CalamariUnavailableError",
    "TranscribeLineFailure",
    "run_calamari_transcribe",
    "run_calamari_transcribe_many",
]
