"""Backward-compatible imports for legacy callers."""

from ...metrics import edit_distance
from .token_metrics import character_error_rate

__all__ = ["character_error_rate", "edit_distance"]
