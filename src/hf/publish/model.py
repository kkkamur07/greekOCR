"""Publish model checkpoints from the Hub staging tree."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.hf.publish.client import PublishClient
from src.hf.publish.staging import (
  ModelStagingRef,
  build_model_card,
  model_staging_dir,
  validate_model_staging,
)


@dataclass(frozen=True, slots=True)
class ModelPublishPlan:
  ref: ModelStagingRef
  namespace: str
  task: str
  repo_id: str
  staging_dir: Path
  registry_model_id: str
  weights_source: str
  dry_run: bool


def plan_model_publish(
  *,
  script: str,
  architecture: str,
  model_version: str,
  registry_tag: str,
  namespace: str,
  task: str,
  registry_model_id: str | None = None,
  staging_root: Path | None = None,
  dry_run: bool = True,
) -> ModelPublishPlan:
  ref = ModelStagingRef(
    script=script,
    architecture=architecture,
    model_version=model_version,
    registry_tag=registry_tag,
  )
  resolved_model_id = registry_model_id or ref.registry_model_id
  staging_dir = model_staging_dir(ref, staging_root=staging_root)
  validate_model_staging(staging_dir, architecture=architecture)
  repo_id = f"{namespace}/{ref.hub_repo_slug}"
  weights_source = ref.weights_source(namespace=namespace)

  return ModelPublishPlan(
    ref=ref,
    namespace=namespace,
    task=task,
    repo_id=repo_id,
    staging_dir=staging_dir,
    registry_model_id=resolved_model_id,
    weights_source=weights_source,
    dry_run=dry_run,
  )


def publish_model(
  plan: ModelPublishPlan,
  *,
  publish_client: PublishClient,
  workspace: Path | None = None,
) -> None:
  if plan.dry_run:
    return

  publish_client.create_repo(plan.repo_id, repo_type="model", private=False)

  if workspace is None:
    raise ValueError("workspace is required for live model publish")

  upload_dir = materialize_model_publish_workspace(plan, workspace)
  revision = publish_client.upload_folder(
    upload_dir,
    plan.repo_id,
    repo_type="model",
    commit_message=(
      f"Publish {plan.registry_model_id}@{plan.ref.registry_tag} "
      f"from Hub staging tree"
    ),
  )
  publish_client.create_tag(
    plan.repo_id,
    tag=plan.ref.registry_tag,
    revision=revision,
    repo_type="model",
  )


def materialize_model_publish_workspace(
  plan: ModelPublishPlan,
  workspace: Path,
) -> Path:
  if workspace.exists():
    shutil.rmtree(workspace)
  shutil.copytree(plan.staging_dir, workspace)
  (workspace / "README.md").write_text(
    build_model_card(
      plan.ref,
      namespace=plan.namespace,
      task=plan.task,
      registry_model_id=plan.registry_model_id,
    ),
    encoding="utf-8",
  )
  return workspace
