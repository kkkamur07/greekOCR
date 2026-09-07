#!/usr/bin/env python3
"""Seed inference models and an optional project-level segment binding.

The editor's model pickers list rows from ``inference_models``, not from
``registry.yaml``, so a registry entry is invisible in dev until this script
puts a row beside it. It seeds the three Calamari transcribe models and the
BLLA segment model, and is an upsert: re-running it after a registry edit
rewrites the existing rows rather than duplicating them.

Task and provider are read out of ``registry.yaml`` rather than restated here.
A catalog row whose ``task`` disagrees with the registry entry its
``artifact_ref`` points at is a row that fails at dispatch, and the only way to
be sure the two agree is to have one of them come from the other.

Binds to the project named by ``BINDING_PROJECT_SLUG`` (default:
byzantine-greek-manuscripts) if it exists. Transcribe bindings are skipped by
default since script-specific models get attached per project when available.

Overrides, both optional and both comma-separated:

* ``DEFAULT_SEGMENT_MODEL`` - the segment row, and the model the project-level
  segment binding points at.
* ``DEFAULT_TRANSCRIBE_MODEL`` - the transcribe rows to seed *instead of* the
  three defaults, for a dev database that should only see one script.
"""

import asyncio
import os

from _bootstrap import ensure_nomikos_on_path
from sqlalchemy import select

ensure_nomikos_on_path()

from backend.ml.infrastructure.orm_models import (  # noqa: E402
    InferenceModel,
    InferenceTask,
    ModelBinding,
)
from backend.project.infrastructure.orm_models import Project  # noqa: E402
from infrastructure import models as _orm_models  # noqa: E402, F401 - register all mappers
from infrastructure.db import system_session  # noqa: E402
from nomikos_inference.contracts.common import RegistryArchitecture  # noqa: E402
from nomikos_inference.registry import load_registry  # noqa: E402


def _ids(env_var: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(env_var, "")
    override = tuple(part.strip() for part in raw.split(",") if part.strip())
    return override or default


SEGMENT_MODELS = _ids("DEFAULT_SEGMENT_MODEL", ("blla-segment",))
TRANSCRIBE_MODELS = _ids(
    "DEFAULT_TRANSCRIBE_MODEL",
    ("greek-calamari-v1", "armenian-calamari-v1", "syriac-calamari-v2"),
)
BINDING_PROJECT_SLUG = os.environ.get(
    "BINDING_PROJECT_SLUG",
    os.environ.get("DEV_ANNOTATED_PROJECT_SLUG", "byzantine-greek-manuscripts"),
)

# ``provider`` is a catalog label, not a dispatch key: nothing reads it back
# except the ordering in ``model_repository.list_models``. It was hardcoded to
# "kraken" for every row, which was already only half true for BLLA (the
# topology is Kraken's, the runtime is our own ONNX graph) and plainly wrong for
# a Calamari model. Naming the architecture is the reading that survives someone
# scanning the table to work out what a row runs on.
#
# ``blla`` and ``blla-segment`` are both spellings the enum accepts. registry.yaml
# uses the first; both mean the same runtime, so both map to the same provider.
_PROVIDER_BY_ARCHITECTURE: dict[RegistryArchitecture, str] = {
    RegistryArchitecture.blla: "kraken",
    RegistryArchitecture.blla_segment: "kraken",
    RegistryArchitecture.calamari: "calamari",
}


async def _upsert_model(*, name: str, task: InferenceTask) -> InferenceModel:
    """Write one catalog row for a registry model id, creating or rewriting it.

    An id the Registry does not know raises here rather than seeding a row that
    resolves to nothing at dispatch time.
    """
    entry = load_registry().models.get(name)
    if entry is None:
        raise SystemExit(f"{name!r} is not in registry.yaml; add the entry before seeding it")
    if entry.task.value != task.value:
        raise SystemExit(
            f"{name!r} is a {entry.task.value} model in registry.yaml, seeded here as {task.value}"
        )
    provider = _PROVIDER_BY_ARCHITECTURE[entry.architecture]
    artifact_ref = f"registry://{name}?tag=stable"
    default_params = {"device": entry.device.value}

    async with system_session() as session:
        result = await session.execute(select(InferenceModel).where(InferenceModel.name == name))
        model = result.scalar_one_or_none()
        if model is None:
            model = InferenceModel(
                name=name,
                provider=provider,
                task=task,
                artifact_ref=artifact_ref,
                default_params=default_params,
            )
            session.add(model)
        else:
            model.provider = provider
            model.task = task
            model.artifact_ref = artifact_ref
            model.default_params = default_params
        await session.commit()
        await session.refresh(model)
        return model


async def _upsert_project_binding(
    *, project: Project, task: InferenceTask, model: InferenceModel
) -> ModelBinding:
    async with system_session() as session:
        result = await session.execute(
            select(ModelBinding).where(
                ModelBinding.project_id == project.id,
                ModelBinding.document_id.is_(None),
                ModelBinding.document_part_id.is_(None),
                ModelBinding.task == task,
            )
        )
        binding = result.scalar_one_or_none()
        if binding is None:
            binding = ModelBinding(
                project_id=project.id,
                task=task,
                model_id=model.id,
                overrides={},
            )
            session.add(binding)
        else:
            binding.model_id = model.id
            binding.overrides = {}
        await session.commit()
        await session.refresh(binding)
        return binding


async def main() -> None:
    segment_models = [
        await _upsert_model(name=name, task=InferenceTask.segment) for name in SEGMENT_MODELS
    ]
    transcribe_models = [
        await _upsert_model(name=name, task=InferenceTask.transcribe) for name in TRANSCRIBE_MODELS
    ]

    for model in segment_models:
        print(f"Seeded segment model: {model.name} ({model.provider}) -> {model.artifact_ref}")
    for model in transcribe_models:
        print(f"Seeded transcribe model: {model.name} ({model.provider}) -> {model.artifact_ref}")
    print(
        "Note: no project-level transcribe binding "
        "(attach script-specific models per project when available)."
    )

    async with system_session() as session:
        project = (
            await session.execute(select(Project).where(Project.slug == BINDING_PROJECT_SLUG))
        ).scalar_one_or_none()

    if project is None:
        print(
            f"No project slug={BINDING_PROJECT_SLUG!r} yet - "
            "run annotated seed first, then re-run this script for bindings."
        )
        return

    # The project-level segment binding still points at one model, the first
    # seeded. Transcribe stays unbound on purpose: three scripts are now in the
    # catalog and picking one of them as a workspace default here would put a
    # Greek model in front of a Syriac manuscript.
    segment_binding = await _upsert_project_binding(
        project=project,
        task=InferenceTask.segment,
        model=segment_models[0],
    )
    print(f"Segment binding on {project.slug} ({project.id}): {segment_binding.id}")


if __name__ == "__main__":
    asyncio.run(main())
