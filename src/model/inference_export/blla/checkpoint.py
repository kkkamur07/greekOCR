"""Load the native BLLA safetensors checkpoint into the Torch graph.

This is ``_load_blla_model`` as it stood in
``inference/architectures/blla/blla.py`` under ADR 0004, moved here by ADR 0006
along with the graph it fills. It is export-time code now: the runtime loads
``blla.onnx`` and never opens the checkpoint.

safetensors carries tensors only and cannot execute code on load, which is why
it is the published native format rather than a pickle - the exporter reads it
on a maintainer's machine, but that is the same file a researcher would have
been fetching under ADR 0004.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch
from src.model.inference_export.blla.model import BLLATorchModel


class BLLACheckpointError(RuntimeError):
    """Raised when a native BLLA checkpoint cannot be used."""


def _validate_checkpoint(checkpoint: object) -> Mapping[str, object]:
    if not isinstance(checkpoint, Mapping):
        raise BLLACheckpointError("BLLA checkpoint must be a mapping")
    if checkpoint.get("format") != "blla-pytorch-v1":
        raise BLLACheckpointError("unsupported BLLA checkpoint format")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise BLLACheckpointError("BLLA checkpoint has no model state dictionary")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise BLLACheckpointError("BLLA checkpoint has an invalid model state dictionary")
    return checkpoint


@lru_cache(maxsize=4)
def load_blla_model(model_path: str) -> BLLATorchModel:
    """Open a safetensors checkpoint and return the graph in eval mode."""
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
    except BLLACheckpointError:
        raise
    except Exception as error:
        raise BLLACheckpointError("unable to safely load BLLA checkpoint") from error
    model.eval()
    return model


__all__ = ["BLLACheckpointError", "load_blla_model"]
