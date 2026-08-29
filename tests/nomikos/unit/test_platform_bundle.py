"""Regression tests for Vercel platform bundle artifact exclusions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.fixtures.paths import REPO_ROOT

BUILD_SCRIPT = REPO_ROOT / "infrastructure" / "platform" / "build.sh"


def test_platform_bundle_excludes_env_files_from_all_copied_trees(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "bundle"

    _write_source_file(source_root, "nomikos/backend/main.py")
    _write_source_file(source_root, "nomikos/backend/.env")
    _write_source_file(source_root, "nomikos/backend/.env.local")
    _write_source_file(source_root, "nomikos/backend/media/private-page.webp")
    _write_source_file(source_root, "nomikos/backend/__pycache__/main.cpython-311.pyc")
    _write_source_file(source_root, "nomikos/infrastructure/alembic.ini")
    _write_source_file(source_root, "nomikos/infrastructure/.env")
    _write_source_file(source_root, "nomikos/infrastructure/.env.production")
    _write_source_file(source_root, "nomikos/infrastructure/__pycache__/settings.cpython-311.pyc")
    _write_source_file(source_root, "nomikos_inference/__init__.py")
    _write_source_file(source_root, "nomikos_inference/admission.py")
    _write_source_file(source_root, "nomikos_inference/registry.yaml")
    _write_source_file(source_root, "nomikos_inference/contracts/__init__.py")
    _write_source_file(source_root, "nomikos_inference/contracts/.env")
    _write_source_file(
        source_root, "nomikos_inference/contracts/__pycache__/common.cpython-311.pyc"
    )
    _write_source_file(source_root, "nomikos_inference/settings.py")
    _write_source_file(source_root, "nomikos_inference/registry/__init__.py")
    _write_source_file(source_root, "nomikos_inference/registry/.env.production")
    _write_source_file(
        source_root, "nomikos_inference/registry/__pycache__/resolve.cpython-311.pyc"
    )
    _write_source_file(source_root, "nomikos_inference/weights/production.pt")
    _write_source_file(source_root, "src/experiments/notebook.ipynb")
    _write_source_file(source_root, "data/local-page.webp")
    _write_source_file(source_root, "nomikos/VERSION")

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
        "nomikos_inference/__init__.py",
        "nomikos_inference/admission.py",
        "nomikos_inference/contracts/__init__.py",
        "nomikos_inference/settings.py",
        "nomikos_inference/registry.yaml",
        "nomikos_inference/registry/__init__.py",
        "nomikos/VERSION",
        "nomikos/backend/main.py",
        "nomikos/infrastructure/alembic.ini",
    }
    assert not [path for path in destination.rglob("*") if path.name == "__pycache__"]


def _write_source_file(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("harmless test sentinel\n")
