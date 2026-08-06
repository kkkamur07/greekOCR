"""Shared paths under the Hub integration root.

Repository-relative only. The **Hub cache** root is not here: it belongs to the
runtime, ships in the published package, and now defaults under the
researcher's home directory - see ``inference.hub.cache.default_cache_root``.
"""

from __future__ import annotations

from pathlib import Path

HF_ROOT = Path(__file__).resolve().parent
DEFAULT_STAGING_ROOT = HF_ROOT / "staging"
DEFAULT_COLLECTION_PATH = HF_ROOT / "publish" / "collection.yaml"
