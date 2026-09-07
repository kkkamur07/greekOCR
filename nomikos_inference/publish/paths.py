"""Shared paths under the publish-side root.

Repository-relative only. This tree sits inside `nomikos_inference/` for import
reasons, not because it ships: `pyproject.toml` holds `nomikos_inference/publish`
out of both build targets, so a researcher who installs the wheel gets none of
it. That is also why every path here is anchored on `__file__` rather than on a
search that could quietly succeed inside site-packages.

The **Hub cache** root is not here: it belongs to the runtime, ships in the
published package, and defaults under the researcher's home directory - see
``nomikos_inference.hub.cache.default_cache_root``.
"""

from __future__ import annotations

from pathlib import Path

PUBLISH_ROOT = Path(__file__).resolve().parent
# The blob trees sit one level down from the Python modules so that the **Hub
# staging tree** directory and `staging.py`, the module that validates it, are
# not two entries with the same name in the same package.
ARTIFACTS_ROOT = PUBLISH_ROOT / "artifacts"
DEFAULT_STAGING_ROOT = ARTIFACTS_ROOT / "staging"
DEFAULT_LOCAL_BUNDLED_ROOT = ARTIFACTS_ROOT / "local"
DEFAULT_COLLECTION_PATH = PUBLISH_ROOT / "collection.yaml"
