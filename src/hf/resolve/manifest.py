"""Hub cache manifest read/write."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FILENAME = ".hub-manifest.json"


@dataclass(frozen=True, slots=True)
class HubCacheManifest:
  repo_id: str
  hub_revision: str
  artifact_path: str
  artifact_sha256: str

  def to_dict(self) -> dict[str, str]:
    return {
      "repo_id": self.repo_id,
      "hub_revision": self.hub_revision,
      "artifact_path": self.artifact_path,
      "artifact_sha256": self.artifact_sha256,
    }

  @classmethod
  def from_dict(cls, data: dict[str, object]) -> HubCacheManifest:
    return cls(
      repo_id=str(data["repo_id"]),
      hub_revision=str(data["hub_revision"]),
      artifact_path=str(data["artifact_path"]),
      artifact_sha256=str(data["artifact_sha256"]),
    )


def manifest_path(cache_dir: Path) -> Path:
  return cache_dir / MANIFEST_FILENAME


def load_manifest(cache_dir: Path) -> HubCacheManifest | None:
  path = manifest_path(cache_dir)
  if not path.is_file():
    return None
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    return HubCacheManifest.from_dict(data)
  except (OSError, TypeError, ValueError, KeyError):
    return None


def save_manifest(cache_dir: Path, manifest: HubCacheManifest) -> None:
  cache_dir.mkdir(parents=True, exist_ok=True)
  manifest_path(cache_dir).write_text(
    json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def manifest_matches_expected(
  manifest: HubCacheManifest,
  *,
  repo_id: str,
  hub_revision: str,
  artifact_sha256: str,
) -> bool:
  return (
    manifest.repo_id == repo_id
    and manifest.hub_revision == hub_revision
    and manifest.artifact_sha256 == artifact_sha256
  )
