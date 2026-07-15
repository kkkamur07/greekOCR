#!/usr/bin/env python3
"""Seed inference models and optional project-level segment bindings.

Does not create a separate "Dev inference defaults" project. When
``BINDING_PROJECT_SLUG`` (default: byzantine-greek-manuscripts) exists,
attaches a project-level segment binding. Transcribe bindings are skipped by
default so script-specific models are attached per project when available.
"""

import asyncio
import os

from sqlalchemy import select

from _bootstrap import ensure_nomicous_on_path

ensure_nomicous_on_path()

from backend.ml.infrastructure.orm_models import (  # noqa: E402
    InferenceModel,
    InferenceTask,
    ModelBinding,
)
from backend.project.infrastructure.orm_models import Project  # noqa: E402

from infrastructure import models as _orm_models  # noqa: E402, F401 - register all mappers
from infrastructure.db import system_session  # noqa: E402

DEFAULT_SEGMENT_MODEL = os.environ.get("DEFAULT_SEGMENT_MODEL", "kraken-segment")
DEFAULT_TRANSCRIBE_MODEL = os.environ.get("DEFAULT_TRANSCRIBE_MODEL", "syriac-calamari-v1")
BINDING_PROJECT_SLUG = os.environ.get(
    "BINDING_PROJECT_SLUG",
    os.environ.get("DEV_ANNOTATED_PROJECT_SLUG", "byzantine-greek-manuscripts"),
)


async def _upsert_model(
    *,
    name: str,
    task: InferenceTask,
    artifact_ref: str,
    default_params: dict,
) -> InferenceModel:
    async with system_session() as session:
        result = await session.execute(select(InferenceModel).where(InferenceModel.name == name))
        model = result.scalar_one_or_none()
        if model is None:
            model = InferenceModel(
                name=name,
                provider="kraken",
                task=task,
                artifact_ref=artifact_ref,
                default_params=default_params,
            )
            session.add(model)
        else:
            model.provider = "kraken"
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
    segment_model = await _upsert_model(
        name=DEFAULT_SEGMENT_MODEL,
        task=InferenceTask.segment,
        artifact_ref=f"registry://{DEFAULT_SEGMENT_MODEL}?tag=stable",
        default_params={"device": "cpu"},
    )
    transcribe_model = await _upsert_model(
        name=DEFAULT_TRANSCRIBE_MODEL,
        task=InferenceTask.transcribe,
        artifact_ref=f"registry://{DEFAULT_TRANSCRIBE_MODEL}?tag=stable",
        default_params={"device": "cpu"},
    )

    print(f"Seeded segment model: {segment_model.name} -> {segment_model.artifact_ref}")
    print(f"Seeded transcribe model: {transcribe_model.name} -> {transcribe_model.artifact_ref}")
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

    segment_binding = await _upsert_project_binding(
        project=project,
        task=InferenceTask.segment,
        model=segment_model,
    )
    print(f"Segment binding on {project.slug} ({project.id}): {segment_binding.id}")


if __name__ == "__main__":
    asyncio.run(main())
