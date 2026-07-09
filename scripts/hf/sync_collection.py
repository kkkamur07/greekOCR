#!/usr/bin/env python3
"""Sync src/hf/publish/collection.yaml to the Hugging Face Hub collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Sync the nomos Hub collection from src/hf/publish/collection.yaml.",
  )
  parser.add_argument(
    "--collection-path",
    type=Path,
    default=REPO_ROOT / "src" / "hf" / "publish" / "collection.yaml",
    help="Path to collection.yaml",
  )
  parser.add_argument(
    "--upload",
    action="store_true",
    help="Perform a live Hub sync (also set HF_PUBLISH=1)",
  )
  return parser.parse_args()


def main() -> int:
  args = _parse_args()

  from src.hf.publish import get_default_publish_client, upload_enabled
  from src.hf.publish import plan_collection_sync, sync_collection

  dry_run = not upload_enabled(upload_flag=args.upload)
  try:
    plan = plan_collection_sync(
      collection_path=args.collection_path,
      dry_run=dry_run,
    )
  except ValueError as exc:
    print(str(exc), file=sys.stderr)
    return 1

  print(f"collection: {plan.collection_slug}")
  print(f"title: {plan.spec.title}")
  for item in plan.spec.models:
    print(f"model: {plan.spec.repo_id(item.slug)}")
  for item in plan.spec.datasets:
    print(f"dataset: {plan.spec.repo_id(item.slug)}")

  if dry_run:
    print("dry-run: no Hub collection sync performed", file=sys.stderr)
    return 0

  sync_collection(plan, publish_client=get_default_publish_client())
  print(f"synced collection {plan.collection_slug}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
