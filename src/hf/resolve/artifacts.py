"""Locate architecture-native Hub artifacts inside a cache directory."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def verify_artifact_sha256(path: Path, expected_sha256: str) -> None:
  actual_sha256 = sha256_file(path)
  if actual_sha256 != expected_sha256:
    raise ValueError(
      f"artifact SHA-256 mismatch for {path}: "
      f"expected {expected_sha256}, got {actual_sha256}"
    )


def find_hub_artifact(cache_dir: Path, *, architecture: str | None) -> Path:
  if architecture == "calamari":
    for name in ("best.pt", "stable.pt"):
      candidate = cache_dir / name
      if candidate.is_file():
        return candidate
    for path in sorted(cache_dir.glob("*.pt")):
      if path.is_file():
        return path
    for name in ("best.ckpt", "stable.ckpt"):
      candidate = cache_dir / name
      if candidate.exists():
        return candidate
    for path in sorted(cache_dir.glob("*.ckpt")):
      if path.is_dir() or path.is_file():
        return path

  if architecture in (None, "kraken-segment", "kraken_segment"):
    for path in sorted(cache_dir.glob("*.mlmodel")):
      if path.is_file():
        return path

  raise FileNotFoundError(
    f"no supported Hub artifact found in cache directory: {cache_dir}"
  )
