#!/usr/bin/env python3
"""Publish a staged dataset to a Hub dataset repo."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Publish a staged dataset to a Hub dataset repo.",
  )
  parser.add_argument(
    "dataset_slug",
    help="Hub dataset slug (e.g. greek-manuscript-lines)",
  )
  parser.add_argument("--script", required=True, help="Script the corpus targets (e.g. greek)")
  parser.add_argument(
    "--namespace",
    default="nomicous",
    help="Hub namespace (default: nomicous)",
  )
  parser.add_argument(
    "--staging-root",
    type=Path,
    default=REPO_ROOT / "src" / "hf" / "staging",
    help="Hub staging tree root",
  )
  parser.add_argument(
    "--upload",
    action="store_true",
    help="Perform a live Hub upload (also set HF_PUBLISH=1)",
  )
  return parser.parse_args()


def main() -> int:
  args = _parse_args()

  from src.hf.publish import get_default_publish_client, upload_enabled
  from src.hf.publish import plan_dataset_publish, publish_dataset
  from src.hf.publish import build_dataset_readme

  dry_run = not upload_enabled(upload_flag=args.upload)
  try:
    plan = plan_dataset_publish(
      dataset_slug=args.dataset_slug,
      namespace=args.namespace,
      script=args.script,
      staging_root=args.staging_root,
      dry_run=dry_run,
    )
  except ValueError as exc:
    print(str(exc), file=sys.stderr)
    return 1

  print(f"staging: {plan.staging_dir}")
  print(f"repo: {plan.repo_id}")
  print(build_dataset_readme(
    plan.ref,
    namespace=plan.namespace,
    script=plan.script,
  ))

  if dry_run:
    print("dry-run: no Hub upload performed", file=sys.stderr)
    return 0

  with tempfile.TemporaryDirectory(prefix="hf-publish-dataset-") as tmp:
    publish_dataset(
      plan,
      publish_client=get_default_publish_client(),
      workspace=Path(tmp),
    )

  print(f"published {plan.repo_id}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
