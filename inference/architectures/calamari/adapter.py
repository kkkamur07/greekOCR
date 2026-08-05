"""Calamari OCR adapter with ONNX and legacy PyTorch dispatch."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from inference.architectures.artifact import ArtifactHandle, resolve_artifact
from inference.architectures.calamari.onnx import (
    CalamariUnavailableError,
    TranscribeLineFailure,
    _decode_greedy,
    _response_from_decoded,
    run_calamari_onnx_transcribe_many,
)
from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)
from inference.architectures.isolation import reraise_if_none_survived
from inference.contracts.transcribe import TranscribeRunResponse

CALAMARI_ARTIFACT_SUFFIXES = frozenset({".pt", ".onnx"})


@lru_cache(maxsize=4)
def _load_checkpoint(
    checkpoint_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[object, list[str] | None, int]:
    """Load the legacy checkpoint only for the transition period."""
    try:
        from src.model.inference_export.calamari.export import load_calamari_checkpoint

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
        unusable_message=(
            f"Calamari runtime requires a self-contained .onnx artifact: {checkpoint_path}"
        ),
        artifact_sha256=artifact_sha256,
    )
    if not line_images:
        raise ValueError("at least one line image is required")

    if checkpoint_path.suffix == ".onnx":
        return _reject_fully_failed_batch(
            run_calamari_onnx_transcribe_many(
                line_images,
                checkpoint_path=checkpoint_path,
            )
        )

    return _reject_fully_failed_batch(
        _run_legacy_pytorch_transcribe_many(
            line_images,
            handle=handle,
        )
    )


def _run_legacy_pytorch_transcribe_many(
    line_images: list[bytes],
    *,
    handle: ArtifactHandle,
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    import torch

    model, charset, line_height = _load_checkpoint(handle.path, handle.fingerprint)
    if not charset:
        raise CalamariUnavailableError(f"Calamari checkpoint has no codec metadata: {handle.path}")

    responses: list[TranscribeRunResponse | TranscribeLineFailure] = []
    with torch.inference_mode():
        for index, image_bytes in enumerate(line_images):
            # Same per-line isolation as the ONNX path: the caller decides what a
            # failed line means, and one of them must not end the batch.
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
