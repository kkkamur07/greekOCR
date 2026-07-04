"""Root inference service package (separate deployable from annote)."""

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

__all__ = [
    "ML_ROOT",
]

