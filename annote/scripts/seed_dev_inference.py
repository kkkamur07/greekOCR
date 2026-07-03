#!/usr/bin/env python3
"""Seed development inference models and project-level bindings."""

import asyncio
import os
import sys

from sqlalchemy import select

# Annote app root on PYTHONPATH when run: PYTHONPATH=. python scripts/seed_dev_inference.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.infrastructure.orm_models import (  # noqa: E402
    InferenceModel,
    InferenceTask,
    ModelBinding,
)
from backend.project.infrastructure.orm_models import Project  # noqa: E402
from backend.users.infrastructure.orm_models import User  # noqa: E402
from infrastructure.db import AsyncSessionLocal  # noqa: E402
from ml.registry import load_registry  # noqa: E402

DEFAULT_SEGMENT_MODEL = os.environ.get("DEFAULT_SEGMENT_MODEL", "kraken-segment-default")
DEFAULT_TRANSCRIBE_MODEL = os.environ.get("DEFAULT_TRANSCRIBE_MODEL", "kraken-transcribe-default")
DEFAULT_SEGMENT_REGISTRY_ID = os.environ.get("DEFAULT_SEGMENT_REGISTRY_ID", "kraken-blla")
DEFAULT_TRANSCRIBE_REGISTRY_ID = os.environ.get(
    "DEFAULT_TRANSCRIBE_REGISTRY_ID", "greek-calamariv1"
)
DEFAULT_REGISTRY_TAG = os.environ.get("DEFAULT_REGISTRY_TAG", "stable")
DEV_PROJECT_SLUG = os.environ.get("DEV_INFERENCE_PROJECT_SLUG", "dev-inference")
DEV_PROJECT_NAME = os.environ.get("DEV_INFERENCE_PROJECT_NAME", "Dev inference defaults")
DEV_USER_EMAIL = os.environ.get("DEV_USER_EMAIL", "dev@kalamos.local")


def _validate_registry_entry(*, registry_model_id: str, registry_tag: str) -> None:
    registry = load_registry()
    if registry_model_id not in registry.models:
        known = ", ".join(sorted(registry.models))
        raise ValueError(f"Unknown registry model id {registry_model_id!r}; known: {known}")
    if registry_tag not in registry.models[registry_model_id].versions:
        known = ", ".join(sorted(registry.models[registry_model_id].versions))
        raise ValueError(
            f"Unknown registry tag {registry_tag!r} for {registry_model_id!r}; known: {known}"
        )


async def _upsert_model(
    *,
    name: str,
    task: InferenceTask,
    registry_model_id: str,
    registry_tag: str,
    default_params: dict,
) -> InferenceModel:
    _validate_registry_entry(registry_model_id=registry_model_id, registry_tag=registry_tag)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(InferenceModel).where(InferenceModel.name == name))
        model = result.scalar_one_or_none()
        if model is None:
            model = InferenceModel(
                name=name,
                provider="kraken" if task == InferenceTask.segment else "calamari",
                task=task,
                registry_model_id=registry_model_id,
                registry_tag=registry_tag,
                default_params=default_params,
            )
            session.add(model)
        else:
            model.provider = "kraken" if task == InferenceTask.segment else "calamari"
            model.task = task
            model.registry_model_id = registry_model_id
            model.registry_tag = registry_tag
            model.default_params = default_params
        await session.commit()
        await session.refresh(model)
        return model


async def _ensure_dev_project() -> Project:
    async with AsyncSessionLocal() as session:
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
    async with AsyncSessionLocal() as session:
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
        registry_model_id=DEFAULT_SEGMENT_REGISTRY_ID,
        registry_tag=DEFAULT_REGISTRY_TAG,
        default_params={"device": "cpu"},
    )
    transcribe_model = await _upsert_model(
        name=DEFAULT_TRANSCRIBE_MODEL,
        task=InferenceTask.transcribe,
        registry_model_id=DEFAULT_TRANSCRIBE_REGISTRY_ID,
        registry_tag=DEFAULT_REGISTRY_TAG,
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

    print(
        f"Seeded segment model: {segment_model.name} -> "
        f"{segment_model.registry_model_id}@{segment_model.registry_tag}"
    )
    print(
        f"Seeded transcribe model: {transcribe_model.name} -> "
        f"{transcribe_model.registry_model_id}@{transcribe_model.registry_tag}"
    )
    print(f"Seeded project: {project.slug} ({project.id})")
    print(f"Segment binding: {segment_binding.id}")
    print(f"Transcribe binding: {transcribe_binding.id}")


if __name__ == "__main__":
    asyncio.run(main())
