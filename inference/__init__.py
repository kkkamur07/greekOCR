"""Root inference service package (separate deployable from annote)."""

from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parent

__all__ = [
    "INFERENCE_ROOT",
]
