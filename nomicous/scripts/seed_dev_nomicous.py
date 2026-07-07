#!/usr/bin/env python3
"""Seed all local development data (idempotent).

Runs on Docker API startup so a fresh or wiped Postgres volume still has:
  - dev@example.com / dev-pass-123
  - inference models + dev-inference project
  - Dev annotated corpus (when data/annotated/data is mounted)

Usage (from nomicous/):

  PYTHONPATH=. python scripts/seed_dev_nomicous.py

Environment:
  ANNOTATED_DATA_ROOT        — override annotated data path (see seed_dev_annotated_data.py)
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seed_dev_annotated_data import (  # noqa: E402
    ANNOTATED_DATA_ROOT,
    _configure_logging,
    _print_summary,
    run_seed,
)
from seed_dev_inference import main as seed_inference  # noqa: E402
from seed_dev_user import main as seed_user  # noqa: E402


async def main() -> None:
    print("=== seed_dev_user ===", flush=True)
    await seed_user()

    print("\n=== seed_dev_inference ===", flush=True)
    await seed_inference()

    if not ANNOTATED_DATA_ROOT.is_dir():
        print(
            f"\nSkipping annotated corpus — data root not found: {ANNOTATED_DATA_ROOT}",
            flush=True,
        )
        return

    print("\n=== seed_dev_annotated_data ===", flush=True)
    _configure_logging(verbose=False)
    stats = await run_seed(force=False, import_history=False)
    _print_summary(stats)


if __name__ == "__main__":
    asyncio.run(main())
