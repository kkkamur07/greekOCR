"""Download and cache Hub model weights for inference."""

from nomikos_inference.hub.artifacts import (
    ArtifactIntegrityError,
    find_hub_artifact,
    sha256_file,
    verify_artifact_sha256,
)
from nomikos_inference.hub.cache import (
    DEFAULT_CACHE_ROOT,
    cache_dir_for,
    default_cache_root,
    resolve_hf_weights_source,
)
from nomikos_inference.hub.client import (
    HubClient,
    get_default_hub_client,
    set_default_hub_client,
)
from nomikos_inference.hub.manifest import (
    HubCacheManifest,
    load_manifest,
    manifest_matches_expected,
    save_manifest,
)
from nomikos_inference.hub.uri import HfWeightsUri, parse_hf_weights_uri

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "ArtifactIntegrityError",
    "HubCacheManifest",
    "HubClient",
    "HfWeightsUri",
    "cache_dir_for",
    "default_cache_root",
    "find_hub_artifact",
    "get_default_hub_client",
    "load_manifest",
    "manifest_matches_expected",
    "parse_hf_weights_uri",
    "resolve_hf_weights_source",
    "save_manifest",
    "set_default_hub_client",
    "sha256_file",
    "verify_artifact_sha256",
]
