#!/usr/bin/env python3
"""Seed development inference models and project-level bindings."""

import asyncio
import os
from pathlib import Path

from sqlalchemy import select

from _bootstrap import ensure_nomicous_on_path

ensure_nomicous_on_path()

from backend.ml.infrastructure.orm_models import (  # noqa: E402
    InferenceModel,
    InferenceTask,
    ModelBinding,
)
from backend.project.infrastructure.orm_models import Project  # noqa: E402
from backend.users.infrastructure.orm_models import User  # noqa: E402

from infrastructure import models as _orm_models  # noqa: E402, F401 — register all mappers
from infrastructure.db import system_session  # noqa: E402

DEFAULT_SEGMENT_MODEL = os.environ.get("DEFAULT_SEGMENT_MODEL", "greek-kraken-segment-v1")
DEFAULT_TRANSCRIBE_MODEL = os.environ.get("DEFAULT_TRANSCRIBE_MODEL", "syriac-calamari-v1")
KRAKEN_MODEL_PATH = Path(os.environ.get("KRAKEN_MODEL_PATH", "../model/kraken"))
DEV_PROJECT_SLUG = os.environ.get("DEV_INFERENCE_PROJECT_SLUG", "dev-inference")
DEV_PROJECT_NAME = os.environ.get("DEV_INFERENCE_PROJECT_NAME", "Dev inference defaults")
DEV_USER_EMAIL = os.environ.get("DEV_USER_EMAIL", "dev@example.com")


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


async def _ensure_dev_project() -> Project:
    async with system_session() as session:
        result = await session.execute(select(Project).where(Project.slug == DEV_PROJECT_SLUG))
        project = result.scalar_one_or_none()
        if project is not None:
            return project

        owner_result = await session.execute(select(User).where(User.email == DEV_USER_EMAIL))
        owner = owner_result.scalar_one_or_none()
        project = Project(
            slug=DEV_PROJECT_SLUG,
            name=DEV_PROJECT_NAME,
            guidelines="Development project for default inference model bindings.",
            owner_id=owner.id if owner is not None else None,
        )
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return project


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
        artifact_ref="registry://greek-kraken-segment-v1?tag=stable",
        default_params={"device": "cpu"},
    )
    transcribe_model = await _upsert_model(
        name=DEFAULT_TRANSCRIBE_MODEL,
        task=InferenceTask.transcribe,
        artifact_ref=f"registry://{DEFAULT_TRANSCRIBE_MODEL}?tag=stable",
        default_params={"device": "cpu"},
    )
    project = await _ensure_dev_project()
    segment_binding = await _upsert_project_binding(
        project=project,
        task=InferenceTask.segment,
        model=segment_model,
    )
    transcribe_binding = await _upsert_project_binding(
        project=project,
        task=InferenceTask.transcribe,
        model=transcribe_model,
    )

    print(f"Seeded segment model: {segment_model.name} -> {segment_model.artifact_ref}")
    print(f"Seeded transcribe model: {transcribe_model.name} -> {transcribe_model.artifact_ref}")
    print(f"Seeded project: {project.slug} ({project.id})")
    print(f"Segment binding: {segment_binding.id}")
    print(f"Transcribe binding: {transcribe_binding.id}")


if __name__ == "__main__":
    asyncio.run(main())
