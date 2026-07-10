"""Resolve hf:// weights sources to local Hub cache paths."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from src.hf.paths import DEFAULT_CACHE_ROOT
from src.hf.resolve.artifacts import find_hub_artifact, verify_artifact_sha256
from src.hf.resolve.client import HubClient, _hub_error_message, get_default_hub_client
from src.hf.resolve.manifest import (
  HubCacheManifest,
  load_manifest,
  manifest_matches_expected,
  save_manifest,
)
from src.hf.resolve.uri import parse_hf_weights_uri

_COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def default_cache_root() -> Path:
  return Path(os.environ.get("HF_CACHE_ROOT", DEFAULT_CACHE_ROOT))


def cache_dir_for(
  registry_model_id: str,
  registry_tag: str,
  *,
  cache_root: Path | None = None,
) -> Path:
  root = cache_root or default_cache_root()
  return root / registry_model_id / registry_tag


def _validate_provenance(*, hub_revision: str | None, artifact_sha256: str | None) -> None:
  if not hub_revision or not _COMMIT_SHA_PATTERN.fullmatch(hub_revision):
    raise ValueError(
      "hf weights source requires an immutable 40-character lowercase Hub commit in hub_revision"
    )
  if not artifact_sha256 or not _SHA256_PATTERN.fullmatch(artifact_sha256):
    raise ValueError(
      "hf weights source requires a 64-character lowercase artifact_sha256"
    )


def _snapshot_download(
  client: HubClient,
  repo_id: str,
  revision: str,
  local_dir: Path,
) -> None:
  try:
    client.snapshot_download(repo_id, revision, local_dir)
  except ValueError:
    raise
  except Exception as exc:
    raise ValueError(_hub_error_message(exc, repo_id=repo_id, revision=revision)) from exc


def resolve_hf_weights_source(
  uri: str,
  *,
  registry_model_id: str,
  registry_tag: str,
  hub_revision: str | None,
  artifact_sha256: str | None,
  architecture: str | None = None,
  hub_client: HubClient | None = None,
  cache_root: Path | None = None,
) -> Path:
  parsed = parse_hf_weights_uri(uri)
  if parsed.registry_tag != registry_tag:
    raise ValueError(
      f"hf weights source registry tag {parsed.registry_tag!r} "
      f"does not match requested registry tag {registry_tag!r}"
    )
  _validate_provenance(
    hub_revision=hub_revision,
    artifact_sha256=artifact_sha256,
  )
  assert hub_revision is not None
  assert artifact_sha256 is not None

  client = hub_client or get_default_hub_client()
  resolved_cache_root = cache_root or default_cache_root()
  cache_dir = cache_dir_for(registry_model_id, registry_tag, cache_root=resolved_cache_root)
  manifest = load_manifest(cache_dir)

  if manifest is not None and manifest_matches_expected(
    manifest,
    repo_id=parsed.repo_id,
    hub_revision=hub_revision,
    artifact_sha256=artifact_sha256,
  ):
    try:
      artifact = find_hub_artifact(cache_dir, architecture=architecture)
      if str(artifact.relative_to(cache_dir)) != manifest.artifact_path:
        raise ValueError("cached Hub artifact path does not match its manifest")
      verify_artifact_sha256(artifact, artifact_sha256)
      return artifact
    except (FileNotFoundError, ValueError):
      pass

  if cache_dir.exists():
    shutil.rmtree(cache_dir)

  try:
    _snapshot_download(client, parsed.repo_id, hub_revision, cache_dir)
    artifact = find_hub_artifact(cache_dir, architecture=architecture)
    verify_artifact_sha256(artifact, artifact_sha256)
    save_manifest(
      cache_dir,
      HubCacheManifest(
        repo_id=parsed.repo_id,
        hub_revision=hub_revision,
        artifact_path=str(artifact.relative_to(cache_dir)),
        artifact_sha256=artifact_sha256,
      ),
    )
    return artifact
  except Exception:
    if cache_dir.exists():
      shutil.rmtree(cache_dir)
    raise
