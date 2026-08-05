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


class ArtifactIntegrityError(ValueError):
  """A weights artifact does not match its pinned SHA-256 digest.

  Subclasses ``ValueError`` for backwards compatibility, but HTTP surfaces
  must map it to a service error (503), never a client error (422): the
  request was fine, the artifact on disk is not.
  """


# Digest of artifacts already hashed in this process, keyed by identity.
# Bounded so a long-lived process cannot grow it without limit.
_VERIFIED_DIGESTS: dict[tuple[str, int, int], str] = {}
_VERIFIED_DIGESTS_MAX = 64


def _artifact_identity(path: Path) -> tuple[str, int, int]:
  """Identify file *content* well enough to reuse a digest.

  Any write to the artifact changes its size or its mtime, so a rewritten,
  truncated, or swapped file misses the cache and gets hashed again.
  """
  stat = path.stat()
  return (str(path), stat.st_size, stat.st_mtime_ns)


def verify_artifact_sha256(path: Path, expected_sha256: str) -> None:
  """Raise ``ArtifactIntegrityError`` unless ``path`` matches its pinned digest.

  The same artifact is verified from several call sites per run (weights
  resolution, the helper capability document, and the architecture adapter).
  Hashing is memoized per ``(path, size, mtime_ns)`` so those cost one read of
  the file rather than one each; a file that changed on disk is always re-read.
  """
  # stat() first: a missing artifact must still raise FileNotFoundError here,
  # exactly as the unmemoized read did.
  identity = _artifact_identity(path)
  if _VERIFIED_DIGESTS.get(identity) == expected_sha256:
    return

  actual_sha256 = sha256_file(path)
  if actual_sha256 != expected_sha256:
    _VERIFIED_DIGESTS.pop(identity, None)
    raise ArtifactIntegrityError(
      f"artifact SHA-256 mismatch for {path}: "
      f"expected {expected_sha256}, got {actual_sha256}"
    )
  if len(_VERIFIED_DIGESTS) >= _VERIFIED_DIGESTS_MAX:
    _VERIFIED_DIGESTS.clear()
  _VERIFIED_DIGESTS[identity] = actual_sha256


def find_hub_artifact(cache_dir: Path, *, architecture: str | None) -> Path:
  if architecture == "calamari":
    # Prefer the self-contained ONNX artifact over the legacy Torch formats.
    for name in ("model.onnx", "best.onnx", "stable.onnx"):
      candidate = cache_dir / name
      if candidate.is_file():
        return candidate
    for path in sorted(cache_dir.glob("*.onnx")):
      if path.is_file():
        return path
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

  if architecture in ("blla", "blla-segment", "blla_segment"):
    # Prefer the Torch-free ONNX artifact over the native safetensors one.
    candidate = cache_dir / "blla.onnx"
    if candidate.is_file():
      return candidate
    for path in sorted(cache_dir.glob("*.onnx")):
      if path.is_file():
        return path

  if architecture in (None, "blla-segment", "kraken_segment"):
    for path in sorted(cache_dir.glob("*.mlmodel")):
      if path.is_file():
        return path
    candidate = cache_dir / "blla.safetensors"
    if candidate.is_file():
      return candidate

  if architecture in ("blla", "blla-segment", "blla_segment"):
    candidate = cache_dir / "blla.safetensors"
    if candidate.is_file():
      return candidate
    for path in sorted(cache_dir.glob("*.safetensors")):
      if path.is_file():
        return path

  raise FileNotFoundError(
    f"no supported Hub artifact found in cache directory: {cache_dir}"
  )
