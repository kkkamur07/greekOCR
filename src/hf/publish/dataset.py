"""Publish datasets from the Hub staging tree."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.hf.publish.client import PublishClient
from src.hf.publish.staging import (
  DatasetStagingRef,
  build_dataset_readme,
  dataset_staging_dir,
  validate_dataset_staging,
)


@dataclass(frozen=True, slots=True)
class DatasetPublishPlan:
  ref: DatasetStagingRef
  namespace: str
  script: str
  repo_id: str
  staging_dir: Path
  dry_run: bool


def plan_dataset_publish(
  *,
  dataset_slug: str,
  namespace: str,
  script: str,
  staging_root: Path | None = None,
  dry_run: bool = True,
) -> DatasetPublishPlan:
  ref = DatasetStagingRef(dataset_slug=dataset_slug)
  staging_dir = dataset_staging_dir(ref, staging_root=staging_root)
  validate_dataset_staging(staging_dir)
  repo_id = ref.repo_id(namespace=namespace)

  return DatasetPublishPlan(
    ref=ref,
    namespace=namespace,
    script=script,
    repo_id=repo_id,
    staging_dir=staging_dir,
    dry_run=dry_run,
  )


def publish_dataset(
  plan: DatasetPublishPlan,
  *,
  publish_client: PublishClient,
  workspace: Path | None = None,
) -> None:
  if plan.dry_run:
    return

  publish_client.create_repo(plan.repo_id, repo_type="dataset", private=False)

  if workspace is None:
    raise ValueError("workspace is required for live dataset publish")

  upload_dir = materialize_dataset_publish_workspace(plan, workspace)
  publish_client.upload_folder(
    upload_dir,
    plan.repo_id,
    repo_type="dataset",
    commit_message=f"Publish dataset {plan.ref.dataset_slug} from Hub staging tree",
  )


def materialize_dataset_publish_workspace(
  plan: DatasetPublishPlan,
  workspace: Path,
) -> Path:
  if workspace.exists():
    shutil.rmtree(workspace)
  shutil.copytree(plan.staging_dir, workspace)
  (workspace / "README.md").write_text(
    build_dataset_readme(
      plan.ref,
      namespace=plan.namespace,
      script=plan.script,
    ),
    encoding="utf-8",
  )
  return workspace
