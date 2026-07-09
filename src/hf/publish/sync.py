"""Sync the in-repo Hub collection definition to Hugging Face."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.hf.publish.client import PublishClient
from src.hf.publish.collection import CollectionSpec, load_collection_spec


@dataclass(frozen=True, slots=True)
class CollectionSyncPlan:
  spec: CollectionSpec
  collection_slug: str
  dry_run: bool


def plan_collection_sync(
  *,
  collection_path: Path | None = None,
  dry_run: bool = True,
) -> CollectionSyncPlan:
  spec = load_collection_spec(collection_path)
  if not spec.hub_slug and not dry_run:
    raise ValueError(
      "collection hub_slug is required for sync; set it after the first Hub "
      "collection is created (for example nomicous/nomos-<id>)"
    )

  collection_slug = spec.hub_slug or f"{spec.namespace}/{spec.slug}"

  return CollectionSyncPlan(
    spec=spec,
    collection_slug=collection_slug,
    dry_run=dry_run,
  )


def sync_collection(
  plan: CollectionSyncPlan,
  *,
  publish_client: PublishClient,
) -> None:
  if plan.dry_run:
    return

  publish_client.update_collection_metadata(
    plan.collection_slug,
    title=plan.spec.title,
    description=plan.spec.description,
  )

  for item in plan.spec.models:
    publish_client.add_collection_item(
      plan.collection_slug,
      item_id=plan.spec.repo_id(item.slug),
      item_type="model",
      note=item.note,
    )

  for item in plan.spec.datasets:
    publish_client.add_collection_item(
      plan.collection_slug,
      item_id=plan.spec.repo_id(item.slug),
      item_type="dataset",
      note=item.note,
    )
