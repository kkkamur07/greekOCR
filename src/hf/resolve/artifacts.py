"""Locate architecture-native Hub artifacts inside a cache directory."""

from __future__ import annotations

from pathlib import Path


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
