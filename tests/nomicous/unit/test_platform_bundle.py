"""Regression tests for Vercel platform bundle artifact exclusions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.fixtures.paths import REPO_ROOT

BUILD_SCRIPT = REPO_ROOT / "deploy" / "platform" / "build.sh"


def test_platform_bundle_excludes_env_files_from_all_copied_trees(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "bundle"

    _write_source_file(source_root, "nomicous/backend/main.py")
    _write_source_file(source_root, "nomicous/backend/.env")
    _write_source_file(source_root, "nomicous/backend/.env.local")
    _write_source_file(source_root, "nomicous/backend/media/private-page.webp")
    _write_source_file(source_root, "nomicous/backend/__pycache__/main.cpython-311.pyc")
    _write_source_file(source_root, "nomicous/infrastructure/alembic.ini")
    _write_source_file(source_root, "nomicous/infrastructure/.env")
    _write_source_file(source_root, "nomicous/infrastructure/.env.production")
    _write_source_file(source_root, "nomicous/infrastructure/__pycache__/settings.cpython-311.pyc")
    _write_source_file(source_root, "inference/__init__.py")
    _write_source_file(source_root, "inference/registry.yaml")
    _write_source_file(source_root, "inference/contracts/__init__.py")
    _write_source_file(source_root, "inference/contracts/.env")
    _write_source_file(source_root, "inference/contracts/__pycache__/common.cpython-311.pyc")
    _write_source_file(source_root, "inference/registry/__init__.py")
    _write_source_file(source_root, "inference/registry/.env.production")
    _write_source_file(source_root, "inference/registry/__pycache__/resolve.cpython-311.pyc")
    _write_source_file(source_root, "inference/weights/production.pt")
    _write_source_file(source_root, "src/experiments/notebook.ipynb")
    _write_source_file(source_root, "data/local-page.webp")
    _write_source_file(source_root, "nomicous/VERSION")

    subprocess.run(
        ["bash", str(BUILD_SCRIPT)],
        check=True,
        env={
            **os.environ,
            "PLATFORM_BUNDLE_SOURCE_ROOT": str(source_root),
            "PLATFORM_BUNDLE_DEST": str(destination),
        },
    )

    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == {
        "inference/__init__.py",
        "inference/contracts/__init__.py",
        "inference/registry.yaml",
        "inference/registry/__init__.py",
        "nomicous/VERSION",
        "nomicous/backend/main.py",
        "nomicous/infrastructure/alembic.ini",
    }
    assert not [path for path in destination.rglob("*") if path.name == "__pycache__"]


def _write_source_file(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("harmless test sentinel\n")
