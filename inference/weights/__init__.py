"""Resolve a registry entry's **weights source** URI to a file on this machine.

Four schemes, one answer: `hf://` through the **Hub cache**, `package://` out of
an installed distribution, `file://local/...` from a source checkout's `src/hf/`,
and `file://...` relative to the inference tree. Whichever it is, a pinned
`artifact_sha256` is verified before the path is handed back.

Nothing here is a cache layout of its own - `inference/hub/cache.py` owns the one
directory that is written to, under the researcher's `~/.nomikos`. This module
only decides which file the runtime should open.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path, PurePosixPath

INFERENCE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS_ROOT = INFERENCE_ROOT / "weights"
LOCAL_BUNDLED_PREFIX = "local/"
LOCAL_BUNDLED_ROOT_ENV = "NOMIKOS_LOCAL_WEIGHTS_ROOT"


def local_bundled_root() -> Path:
    """Root that ``file://local/...`` **weights source**s resolve against.

    **Local bundled weights** are a source-checkout affordance: they exist for
    offline development and for Docker images that copy `src/hf/` in. They are
    deliberately not shipped inside the published wheel - checkpoints are
    published to the Hub and fetched by `hf://`, digest-verified - so the
    installed package has no `src/hf/` to point at and says so instead of
    resolving to a path that happens to exist inside site-packages.
    """
    override = os.environ.get(LOCAL_BUNDLED_ROOT_ENV)
    if override:
        return Path(override).expanduser()

    checkout_root = INFERENCE_ROOT.parent / "src" / "hf"
    if checkout_root.is_dir():
        return checkout_root

    raise FileNotFoundError(
        "local bundled weights are only available in a source checkout; "
        f"use an hf:// weights source or set {LOCAL_BUNDLED_ROOT_ENV}"
    )


def resolve_weights_source(
    uri: str,
    *,
    inference_root: Path = INFERENCE_ROOT,
    registry_model_id: str | None = None,
    registry_tag: str | None = None,
    hub_revision: str | None = None,
    artifact_sha256: str | None = None,
    architecture: str | None = None,
) -> Path:
    if uri.startswith("hf://"):
        if not registry_model_id or not registry_tag:
            raise ValueError("hf weights source requires registry_model_id and registry_tag")
        from inference.hub import resolve_hf_weights_source

        return resolve_hf_weights_source(
            uri,
            registry_model_id=registry_model_id,
            registry_tag=registry_tag,
            hub_revision=hub_revision,
            artifact_sha256=artifact_sha256,
            architecture=architecture,
        )

    if uri.startswith("package://"):
        package_resource = uri.removeprefix("package://")
        package_name, _, resource_name = package_resource.partition("/")
        if not package_name or not resource_name:
            raise ValueError("package weights source must be package://<package>/<resource>")
        # ``importlib.resources ... joinpath`` follows pathlib's absolute-override
        # semantics: an absolute ``resource_name`` ("/etc/passwd") escapes the
        # package root entirely, and the ``..`` check below never fires because
        # there is no ``..`` segment. Reject both shapes before joining.
        if PurePosixPath(resource_name).is_absolute() or ".." in resource_name.split("/"):
            raise ValueError("package weights source must stay within the package")
        if not artifact_sha256:
            raise ValueError("package weights source requires a pinned artifact_sha256")
        resource = resources.files(package_name).joinpath(resource_name)
        if not resource.is_file():
            raise FileNotFoundError(f"package weights source not found: {uri}")
        resolved_path = Path(str(resource))
        from inference.hub.artifacts import verify_artifact_sha256

        verify_artifact_sha256(resolved_path, artifact_sha256)
        return resolved_path

    if not uri.startswith("file://"):
        raise ValueError(f"unsupported weights source scheme: {uri}")

    relative = uri.removeprefix("file://")
    if not relative:
        raise ValueError("file weights source must name a path")
    source_path = Path(relative)
    if source_path.is_absolute():
        raise ValueError("file weights source must be relative to INFERENCE_ROOT or src/hf/")

    if relative.startswith(LOCAL_BUNDLED_PREFIX):
        resolved_root = local_bundled_root().resolve()
        resolved_path = (resolved_root / source_path).resolve()
        root_label = "the local bundled weights root"
    else:
        resolved_root = inference_root.resolve()
        resolved_path = (resolved_root / source_path).resolve()
        root_label = "INFERENCE_ROOT"

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"file weights source must stay within {root_label}") from exc

    if artifact_sha256:
        from inference.hub.artifacts import verify_artifact_sha256

        verify_artifact_sha256(resolved_path, artifact_sha256)
    return resolved_path
