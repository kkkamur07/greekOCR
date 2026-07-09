#!/usr/bin/env python3
"""Prefetch Hub weights for a registry model id and registry tag."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download registry model weights into the Hub cache.",
    )
    parser.add_argument("registry_model_id", help="Registry model id (e.g. greek-calamari-v1)")
    parser.add_argument(
        "--registry-tag",
        default="stable",
        help="Registry tag to resolve (default: stable)",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REPO_ROOT / "inference" / "registry.yaml",
        help="Path to registry.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from inference.registry import get_model_entry, load_registry
    from inference.weights import resolve_weights_source

    registry = load_registry(args.registry_path)
    try:
        entry = get_model_entry(registry, args.registry_model_id, args.registry_tag)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    weights_source = entry.versions[args.registry_tag].weights_source
    if not weights_source.startswith("hf://"):
        print(
            f"{args.registry_model_id}@{args.registry_tag} uses {weights_source!r}; "
            "fetch_model only warms hf:// weights sources.",
            file=sys.stderr,
        )
        return 1

    try:
        artifact = resolve_weights_source(
            weights_source,
            registry_model_id=args.registry_model_id,
            registry_tag=args.registry_tag,
            architecture=entry.architecture.value,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
