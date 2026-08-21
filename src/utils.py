"""Conventions for mutable runtime artifacts outside the source package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved artifact locations for one engine and run."""

    run_dir: Path
    log_file: Path


def runtime_paths(*, root: str | Path, engine: str, run_name: str) -> RuntimePaths:
    """Place model artifacts and logs in separate trees under ``root``."""
    root_path = Path(root).expanduser().resolve()
    run_dir = root_path / "runs" / engine / run_name
    log_file = root_path / "logs" / engine / f"{run_name}.log"
    return RuntimePaths(run_dir=run_dir, log_file=log_file)


#! need to remove this