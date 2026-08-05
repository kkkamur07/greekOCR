"""Backward-compatible imports for legacy callers.

Metric implementations are centralized in :mod:`src.models.trocr.metrics`.
"""

from .metrics import character_error_rate, edit_distance

__all__ = ["character_error_rate", "edit_distance"]
