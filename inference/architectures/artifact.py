"""Artifact preflight shared by every architecture execution path.

Each execution path - once four of them, Calamari and BLLA on ONNX and on
Torch, now the two Torch paths ADR 0004 kept - used to open its artifact with
its own copy of the same steps, and the copies had already drifted apart. They
are collapsed here because the *order* of those steps is load-bearing: each
failure says something different about a deployment, so a path that verified
the digest before checking existence, or that raised a bare ``ValueError`` for
an unusable suffix, would describe the same broken deployment differently from
its sibling.

The order, and what each step says:

1. missing file    -> ``FileNotFoundError``                  (weights unavailable)
2. wrong suffix    -> the caller's ``RuntimeError`` subclass (runtime unusable)
3. bad digest      -> ``ArtifactIntegrityError``             (integrity failure)

None of the three is a client error: in all three cases the request was fine
and the artifact on disk is not. Step 3 is the subtle one -
``ArtifactIntegrityError`` subclasses ``ValueError``, so anything that sorts
these by type has to check it before its ``ValueError`` branch. Verifying the
digest before the suffix check would not change that, but verifying it before
the existence check would turn a missing file into an ``OSError`` from the
hasher.

Step 3 also gates a code-execution surface: ``artifact_sha256`` is verified
here, before the architecture loader ever opens the file, so a Calamari
``.pt`` checkpoint is never handed to ``torch.load`` unverified.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from inference.hub.artifacts import verify_artifact_sha256


@dataclass(frozen=True)
class ArtifactHandle:
    """A verified artifact, in the shape an ``lru_cache``d loader wants.

    ``path`` is a ``str`` and ``fingerprint`` a plain tuple because both are
    cache keys: ``Path`` hashes fine but compares by value across processes,
    and the fingerprint is what makes a *replaced* artifact file miss the
    cache instead of serving the previous model for the life of the process.
    """

    path: str
    fingerprint: tuple[int, int]


def artifact_fingerprint(path: Path) -> tuple[int, int]:
    """Cache-key component so replaced artifact files are reloaded."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def resolve_artifact(
    path: Path,
    *,
    label: str,
    allowed_suffixes: Collection[str],
    unusable_error: type[RuntimeError],
    unusable_message: str,
    artifact_sha256: str | None = None,
) -> ArtifactHandle:
    """Check an artifact is present, loadable by this runtime, and intact.

    ``label`` names the artifact in the not-found message; ``unusable_error``
    and ``unusable_message`` stay per-architecture because they say which
    artifact format *that* runtime can load, which is exactly the part no
    shared code can know. Everything else - the order, the exception types,
    and therefore the HTTP statuses - is fixed here for all of them.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.suffix not in allowed_suffixes:
        raise unusable_error(unusable_message)
    if artifact_sha256:
        verify_artifact_sha256(path, artifact_sha256)
    return ArtifactHandle(path=str(path), fingerprint=artifact_fingerprint(path))


__all__ = [
    "ArtifactHandle",
    "artifact_fingerprint",
    "resolve_artifact",
]
