"""Shared paths under the Hub integration root."""

from __future__ import annotations

from pathlib import Path

HF_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = HF_ROOT / "cache"
DEFAULT_STAGING_ROOT = HF_ROOT / "staging"
DEFAULT_COLLECTION_PATH = HF_ROOT / "publish" / "collection.yaml"
