"""Allocate unique, non-destructive directories for training runs."""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def allocate_run_directory(base_dir: Path, name: str) -> Path:
    """Reserve and return a new directory below ``base_dir``.

    A training invocation must never reuse an existing directory: checkpoints,
    metrics, and resolved configuration are all mutable during training.  The
    Slurm job ID makes cluster paths easy to identify; local invocations receive
    a random suffix instead.
    """
    safe_name = _UNSAFE_NAME.sub("-", name).strip("-.") or "training"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_suffix = _run_suffix()
    base_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(100):
        suffix = run_suffix if attempt == 0 else f"{run_suffix}-{uuid.uuid4().hex[:8]}"
        run_dir = base_dir / f"{safe_name}--{timestamp}--{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir

    raise RuntimeError(f"Unable to allocate a unique training directory under {base_dir}.")


def _run_suffix() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        return f"slurm-{job_id}" if task_id is None else f"slurm-{job_id}-{task_id}"
    return f"local-{uuid.uuid4().hex[:8]}"
