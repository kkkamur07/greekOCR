"""Calamari transcription on the ONNX Runtime CPU runtime (ADR 0006).

The Hub artifact is ``best.onnx``: the graph carries its own codec, line
height and blank index in ``metadata_props``, so the runtime needs neither the
``.pt`` checkpoint nor a sidecar to decode. ``resolve_artifact`` verifies the
digest before the artifact is opened; ``reraise_if_none_survived`` treats an
all-failed batch as a failed run rather than a page of per-line errors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from inference.architectures.artifact import ArtifactHandle, resolve_artifact
from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)
from inference.architectures.isolation import reraise_if_none_survived
from inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse

# The Calamari **Hub artifact** is the self-contained ONNX graph. There is one
# runtime format per architecture; ``find_hub_artifact`` enforces the same rule
# on the cache directory so a repo holding both formats cannot silently decide
# which runtime ran.
CALAMARI_ARTIFACT_SUFFIXES = frozenset({".onnx"})


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


def _metadata_int(metadata: Mapping[str, str], key: str, *, minimum: int) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as error:
        raise CalamariUnavailableError(f"Calamari ONNX metadata has invalid {key}") from error
    if value < minimum:
        raise CalamariUnavailableError(f"Calamari ONNX metadata has invalid {key}")
    return value


@lru_cache(maxsize=4)
def _load_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[Any, list[str], int]:
    """Open a session and read the codec the graph carries with it.

    ``fingerprint`` is part of the cache key rather than an argument the loader
    reads: it is what makes a *replaced* artifact file miss the cache instead of
    serving the previous model for the life of the process.
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("format") != "calamari-onnx-v1":
            raise CalamariUnavailableError("unsupported Calamari ONNX artifact format")
        classes = _metadata_int(metadata, "classes", minimum=2)
        line_height = _metadata_int(metadata, "line_height", minimum=1)
        if metadata.get("blank_index") != "0":
            raise CalamariUnavailableError("Calamari ONNX artifact has an unsupported blank index")
        # The exporter bakes any positive temperature into the graph itself
        # (``CalamariTorchModel.forward`` divides the logits before tracing), so
        # the session output is already temperature-scaled. The metadata value
        # is validated only to reject corrupted artifacts; the runtime must
        # NOT re-apply it to the logits.
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
    except ImportError as error:
        raise CalamariUnavailableError(
            "onnxruntime is required for the Calamari runtime"
        ) from error
    except Exception as error:
        raise CalamariUnavailableError("unable to load Calamari ONNX artifact") from error


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
    # Checked before the artifact: an empty batch is a client error (422)
    # regardless of the weights on disk, and running the artifact preflight
    # first would report a missing artifact (503) for a request that was
    # never runnable to begin with. See ``architectures.artifact`` for why its
    # own failures are ordered the same way.
    if not line_images:
        raise ValueError("at least one line image is required")

    handle = resolve_artifact(
        checkpoint_path,
        label="Calamari model",
        allowed_suffixes=CALAMARI_ARTIFACT_SUFFIXES,
        unusable_error=CalamariUnavailableError,
        unusable_message=(f"Calamari runtime requires an .onnx model: {checkpoint_path}"),
        artifact_sha256=artifact_sha256,
    )
    return _reject_fully_failed_batch(_run_onnx_transcribe_many(line_images, handle=handle))


def _run_onnx_transcribe_many(
    line_images: list[bytes],
    *,
    handle: ArtifactHandle,
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    session, charset, line_height = _load_session(handle.path, handle.fingerprint)
    if not charset:
        raise CalamariUnavailableError(f"Calamari artifact has no codec metadata: {handle.path}")

    responses: list[TranscribeRunResponse | TranscribeLineFailure] = []
    for index, image_bytes in enumerate(line_images):
        # Per-line isolation: the caller decides what a failed line means,
        # and one of them must not end the batch.
        try:
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
            # Softmax in NumPy rather than in the graph: the exporter traces the
            # logits so the temperature it baked in stays visible to anything
            # comparing against the reference forward.
            logits = np.asarray(outputs[0], dtype=np.float32)[0]
            logits -= np.max(logits, axis=-1, keepdims=True)
            softmax = np.exp(logits)
            softmax /= np.sum(softmax, axis=-1, keepdims=True)
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
