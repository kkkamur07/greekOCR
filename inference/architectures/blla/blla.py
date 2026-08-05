"""BLLA segmentation inference adapter.

The **Hub artifact** is ``blla.safetensors``: safetensors carries tensors only
and cannot execute code on load, which is why it is preferred over a pickled
checkpoint here (ADR 0004). The **artifact SHA-256** is still verified through
``architectures.artifact`` before the file is opened.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from inference.architectures.artifact import resolve_artifact
from inference.architectures.blla.blla_model import BLLATorchModel
from inference.architectures.blla.blla_preprocessing import preprocess_blla_image
from inference.architectures.blla.blla_runtime import build_blla_segment_response
from inference.contracts.segment import SegmentRunResponse

BLLA_ARTIFACT_SUFFIXES = frozenset({".safetensors"})


class BLLAUnavailableError(RuntimeError):
    """Raised when a native BLLA checkpoint cannot be used."""


def _validate_checkpoint(checkpoint: object) -> Mapping[str, object]:
    if not isinstance(checkpoint, Mapping):
        raise BLLAUnavailableError("BLLA checkpoint must be a mapping")
    if checkpoint.get("format") != "blla-pytorch-v1":
        raise BLLAUnavailableError("unsupported BLLA checkpoint format")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise BLLAUnavailableError("BLLA checkpoint has no model state dictionary")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise BLLAUnavailableError("BLLA checkpoint has an invalid model state dictionary")
    return checkpoint


@lru_cache(maxsize=4)
def _load_blla_model(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> BLLATorchModel:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        checkpoint = _validate_checkpoint(
            {
                "format": metadata.get("format"),
                "state_dict": load_file(model_path, device="cpu"),
            }
        )
        model = BLLATorchModel()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except BLLAUnavailableError:
        raise
    except Exception as error:
        raise BLLAUnavailableError("unable to safely load BLLA checkpoint") from error
    model.eval()
    return model


def run_blla_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse:
    """Run native BLLA and return the legacy-compatible segment contract."""

    handle = resolve_artifact(
        model_path,
        label="BLLA model",
        allowed_suffixes=BLLA_ARTIFACT_SUFFIXES,
        unusable_error=BLLAUnavailableError,
        unusable_message=(f"native BLLA runtime requires a safetensors checkpoint: {model_path}"),
        artifact_sha256=artifact_sha256,
    )

    model = _load_blla_model(handle.path, handle.fingerprint)
    # ``_load_blla_model`` already called ``eval()``; the model is cached across
    # calls, so assert it here rather than trust the cache.
    model.eval()
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        prepared = preprocess_blla_image(image)
        with torch.inference_mode():
            logits = model(prepared.tensor.unsqueeze(0))[0].cpu().numpy()
        return build_blla_segment_response(image, logits, prepared, params=params)
