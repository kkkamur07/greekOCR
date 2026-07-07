"""Registry lookup helpers shared by the jobs API and inference runner."""

from __future__ import annotations

from pathlib import Path

from inference.contracts.common import InferenceTask
from inference.registry import RegistryModelEntry, get_model_entry, load_registry


def resolve_registry_entry(
    *,
    registry_model_id: str,
    registry_tag: str,
    task: InferenceTask,
    registry_path: Path | None = None,
) -> RegistryModelEntry:
    registry = load_registry(registry_path)
    entry = get_model_entry(registry, registry_model_id, registry_tag)
    if entry.task != task:
        raise ValueError(
            f"task {task.value!r} does not match registry model task {entry.task.value!r}"
        )
    return entry
