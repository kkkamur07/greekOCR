"""Resolve hf:// weights sources to local Hub cache paths."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.hf.paths import DEFAULT_CACHE_ROOT
from src.hf.resolve.artifacts import find_hub_artifact
from src.hf.resolve.client import HubClient, _hub_error_message, get_default_hub_client
from src.hf.resolve.manifest import (
  HubCacheManifest,
  hash_directory,
  load_manifest,
  manifest_matches_remote,
  save_manifest,
)
from src.hf.resolve.uri import parse_hf_weights_uri


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


def _resolve_revision_sha(
  client: HubClient,
  repo_id: str,
  revision: str,
) -> str:
  try:
    return client.resolve_revision_sha(repo_id, revision)
  except ValueError:
    raise
  except Exception as exc:
    raise ValueError(_hub_error_message(exc, repo_id=repo_id, revision=revision)) from exc


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

  client = hub_client or get_default_hub_client()
  resolved_cache_root = cache_root or default_cache_root()
  cache_dir = cache_dir_for(registry_model_id, registry_tag, cache_root=resolved_cache_root)
  revision = registry_tag
  revision_sha = _resolve_revision_sha(client, parsed.repo_id, revision)
  manifest = load_manifest(cache_dir)

  if manifest is not None and manifest_matches_remote(manifest, revision_sha=revision_sha):
    try:
      return find_hub_artifact(cache_dir, architecture=architecture)
    except FileNotFoundError:
      pass

  if cache_dir.exists():
    shutil.rmtree(cache_dir)

  _snapshot_download(client, parsed.repo_id, revision, cache_dir)
  save_manifest(
    cache_dir,
    HubCacheManifest(
      repo_id=parsed.repo_id,
      revision=revision,
      revision_sha=revision_sha,
      artifact_hash=hash_directory(cache_dir),
    ),
  )
  return find_hub_artifact(cache_dir, architecture=architecture)
