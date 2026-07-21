"""Calamari OCR adapter with ONNX and legacy PyTorch dispatch."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from src.hf.resolve.artifacts import verify_artifact_sha256

from inference.architectures.calamari.onnx import (
    CalamariUnavailableError,
    _decode_greedy,
    _response_from_decoded,
    run_calamari_onnx_transcribe_many,
)
from inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)
from inference.contracts.transcribe import TranscribeRunResponse


def _file_fingerprint(path: Path) -> tuple[int, int]:
    """Cache-key component so replaced artifact files are reloaded."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


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


def run_calamari_transcribe_many(
    line_images: list[bytes],
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None = None,
) -> list[TranscribeRunResponse]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Calamari checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix not in {".pt", ".onnx"}:
        raise CalamariUnavailableError(
            f"Calamari runtime requires a self-contained .onnx artifact: {checkpoint_path}"
        )
    if artifact_sha256:
        verify_artifact_sha256(checkpoint_path, artifact_sha256)
    if not line_images:
        raise ValueError("at least one line image is required")

    if checkpoint_path.suffix == ".onnx":
        return run_calamari_onnx_transcribe_many(
            line_images,
            checkpoint_path=checkpoint_path,
        )

    return _run_legacy_pytorch_transcribe_many(
        line_images,
        checkpoint_path=checkpoint_path,
    )


def _run_legacy_pytorch_transcribe_many(
    line_images: list[bytes],
    *,
    checkpoint_path: Path,
) -> list[TranscribeRunResponse]:
    import torch

    model, charset, line_height = _load_checkpoint(
        str(checkpoint_path), _file_fingerprint(checkpoint_path)
    )
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
    artifact_sha256: str | None = None,
) -> TranscribeRunResponse:
    return run_calamari_transcribe_many(
        [image_bytes],
        checkpoint_path=checkpoint_path,
        artifact_sha256=artifact_sha256,
    )[0]


__all__ = [
    "CalamariUnavailableError",
    "run_calamari_transcribe",
    "run_calamari_transcribe_many",
]
