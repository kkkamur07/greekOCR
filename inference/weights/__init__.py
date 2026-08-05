"""Weight source resolution and server-side cache layout."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS_ROOT = INFERENCE_ROOT / "weights"
LOCAL_BUNDLED_PREFIX = "local/"
LOCAL_BUNDLED_ROOT_ENV = "NOMICOUS_LOCAL_WEIGHTS_ROOT"


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
        resource = resources.files(package_name).joinpath(resource_name)
        if not resource.is_file():
            raise FileNotFoundError(f"package weights source not found: {uri}")
        resolved_path = Path(str(resource))
        if artifact_sha256:
            from inference.hub.artifacts import verify_artifact_sha256

            verify_artifact_sha256(resolved_path, artifact_sha256)
        return resolved_path

    if not uri.startswith("file://"):
        raise ValueError(f"unsupported weights source scheme: {uri}")

    relative = uri.removeprefix("file://")
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
