"""Hugging Face Hub publishing tools and the local/staging weight trees.

Runtime `hf://` resolution no longer lives here: it moved into the published
`nomicous-inference` package as `inference.hub` (ADR 0002), because it is on
the inference path and a runtime that cannot fetch its own weights is not a
runtime. What is left under `src/hf/` is publish-side and repository-relative -
the **Hub staging tree**, **Local bundled weights**, model cards, and
collection sync - none of which ships to a researcher.
"""

from src.hf.paths import DEFAULT_COLLECTION_PATH, DEFAULT_STAGING_ROOT, HF_ROOT

__all__ = [
  "DEFAULT_COLLECTION_PATH",
  "DEFAULT_STAGING_ROOT",
  "HF_ROOT",
]
