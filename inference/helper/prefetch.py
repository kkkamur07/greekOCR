"""Prefetch local-eligible Hub weights on helper startup (not mid-/run)."""

from __future__ import annotations

import logging
from pathlib import Path

from inference.contracts.common import HostEligibility
from inference.registry import load_registry
from inference.weights import resolve_weights_source

logger = logging.getLogger(__name__)


def prefetch_local_eligible_weights(registry_path: Path) -> None:
    """Download default local-eligible hf:// weights into the helper cache.

    Skips package:// / file:// sources (already local) and any entry lacking
    Hub provenance. Failures are logged and do not prevent the helper from serving.
    """
    registry = load_registry(registry_path)
    for model_id, entry in sorted(registry.models.items()):
        if entry.host_eligibility not in {HostEligibility.local, HostEligibility.any}:
            continue
        for tag, version in entry.versions.items():
            if not version.weights_source.startswith("hf://"):
                continue
            if not version.hub_revision or not version.artifact_sha256:
                logger.info(
                    "helper_weight_prefetch_skip model=%s tag=%s reason=missing_provenance",
                    model_id,
                    tag,
                )
                continue
            try:
                path = resolve_weights_source(
                    version.weights_source,
                    registry_model_id=model_id,
                    registry_tag=tag,
                    hub_revision=version.hub_revision,
                    artifact_sha256=version.artifact_sha256,
                    architecture=entry.architecture.value,
                )
                logger.info(
                    "helper_weight_prefetch_ok model=%s tag=%s path=%s",
                    model_id,
                    tag,
                    path,
                )
            except Exception:
                logger.exception(
                    "helper_weight_prefetch_failed model=%s tag=%s",
                    model_id,
                    tag,
                )
