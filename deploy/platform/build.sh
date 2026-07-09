#!/usr/bin/env bash
# Bundle platform API sources into deploy/platform for Vercel.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$(cd "$(dirname "$0")" && pwd)"

rm -rf "$DEST/nomicous" "$DEST/inference"
mkdir -p "$DEST/nomicous" "$DEST/inference"

python - "$ROOT" "$DEST" <<'PY'
import fnmatch
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
dest = Path(sys.argv[2])


def ignore_backend(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if (
            name == "media"
            or name == "__pycache__"
            or name == ".env"
            or name.endswith(".pyc")
            or fnmatch.fnmatch(name, ".env.*")
        ):
            ignored.add(name)
    return ignored


def ignore_python_cache(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


shutil.copytree(
    root / "nomicous" / "backend",
    dest / "nomicous" / "backend",
    ignore=ignore_backend,
)
shutil.copytree(
    root / "nomicous" / "infrastructure",
    dest / "nomicous" / "infrastructure",
    ignore=ignore_python_cache,
)
shutil.copy2(root / "inference" / "__init__.py", dest / "inference" / "__init__.py")
shutil.copy2(root / "inference" / "registry.yaml", dest / "inference" / "registry.yaml")
shutil.copytree(root / "inference" / "contracts", dest / "inference" / "contracts")
shutil.copytree(root / "inference" / "registry", dest / "inference" / "registry")
shutil.copy2(root / "nomicous" / "VERSION", dest / "nomicous" / "VERSION")
PY
