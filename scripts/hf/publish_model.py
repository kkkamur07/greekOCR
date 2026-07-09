#!/usr/bin/env python3
"""Publish inference checkpoints from the Hub staging tree to a Hub model repo."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Publish a staged inference checkpoint to a Hub model repo.",
  )
  parser.add_argument("--script", required=True, help="Script (e.g. greek)")
  parser.add_argument("--architecture", required=True, help="Architecture (e.g. calamari)")
  parser.add_argument("--model-version", required=True, help="Model version (e.g. v1)")
  parser.add_argument(
    "--registry-tag",
    default="stable",
    help="Registry tag / Hub revision to create (default: stable)",
  )
  parser.add_argument(
    "--namespace",
    default="nomicous",
    help="Hub namespace (default: nomicous)",
  )
  parser.add_argument(
    "--task",
    default="transcribe",
    choices=("transcribe", "segment"),
    help="Inference task for the model card (default: transcribe)",
  )
  parser.add_argument(
    "--registry-model-id",
    help="Override registry model id (default: derived from staging path)",
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
  from src.hf.publish import plan_model_publish, publish_model
  from src.hf.publish import build_model_card

  dry_run = not upload_enabled(upload_flag=args.upload)
  try:
    plan = plan_model_publish(
      script=args.script,
      architecture=args.architecture,
      model_version=args.model_version,
      registry_tag=args.registry_tag,
      namespace=args.namespace,
      task=args.task,
      registry_model_id=args.registry_model_id,
      staging_root=args.staging_root,
      dry_run=dry_run,
    )
  except ValueError as exc:
    print(str(exc), file=sys.stderr)
    return 1

  print(f"staging: {plan.staging_dir}")
  print(f"repo: {plan.repo_id}")
  print(f"registry model id: {plan.registry_model_id}")
  print(f"weights source: {plan.weights_source}")
  print(build_model_card(
    plan.ref,
    namespace=plan.namespace,
    task=plan.task,
    registry_model_id=plan.registry_model_id,
  ))

  if dry_run:
    print("dry-run: no Hub upload performed", file=sys.stderr)
    return 0

  with tempfile.TemporaryDirectory(prefix="hf-publish-model-") as tmp:
    publish_model(
      plan,
      publish_client=get_default_publish_client(),
      workspace=Path(tmp),
    )

  print(f"published {plan.repo_id}@{plan.ref.registry_tag}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
