"""Hub cache manifest read/write."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FILENAME = ".hub-manifest.json"


@dataclass(frozen=True, slots=True)
class HubCacheManifest:
  repo_id: str
  revision: str
  revision_sha: str
  artifact_hash: str

  def to_dict(self) -> dict[str, str]:
    return {
      "repo_id": self.repo_id,
      "revision": self.revision,
      "revision_sha": self.revision_sha,
      "artifact_hash": self.artifact_hash,
    }

  @classmethod
  def from_dict(cls, data: dict[str, object]) -> HubCacheManifest:
    return cls(
      repo_id=str(data["repo_id"]),
      revision=str(data["revision"]),
      revision_sha=str(data["revision_sha"]),
      artifact_hash=str(data["artifact_hash"]),
    )


def manifest_path(cache_dir: Path) -> Path:
  return cache_dir / MANIFEST_FILENAME


def load_manifest(cache_dir: Path) -> HubCacheManifest | None:
  path = manifest_path(cache_dir)
  if not path.is_file():
    return None
  data = json.loads(path.read_text(encoding="utf-8"))
  return HubCacheManifest.from_dict(data)


def save_manifest(cache_dir: Path, manifest: HubCacheManifest) -> None:
  cache_dir.mkdir(parents=True, exist_ok=True)
  manifest_path(cache_dir).write_text(
    json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def manifest_matches_remote(manifest: HubCacheManifest, *, revision_sha: str) -> bool:
  return manifest.revision_sha == revision_sha


def hash_directory(root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_FILENAME):
    digest.update(str(path.relative_to(root)).encode("utf-8"))
    digest.update(path.read_bytes())
  return digest.hexdigest()
