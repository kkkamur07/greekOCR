#!/usr/bin/env bash
# Bundle platform API sources into deploy/platform for Vercel.
set -euo pipefail

ROOT="${PLATFORM_BUNDLE_SOURCE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
DEST="${PLATFORM_BUNDLE_DEST:-$(cd "$(dirname "$0")" && pwd)}"

if [[ "$DEST" == "/" ]]; then
    echo "Refusing to use the filesystem root as a platform bundle destination." >&2
    exit 1
fi

rm -rf "$DEST/nomicous" "$DEST/inference"
mkdir -p "$DEST/nomicous" "$DEST/inference"

python - "$ROOT" "$DEST" <<'PY'
import fnmatch
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
dest = Path(sys.argv[2])


def ignore_deploy_artifacts(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if (
            name == "__pycache__"
            or name == ".env"
            or name.endswith(".pyc")
            or fnmatch.fnmatch(name, ".env.*")
        ):
            ignored.add(name)
    return ignored


def ignore_backend(directory: str, names: list[str]) -> set[str]:
    return ignore_deploy_artifacts(directory, names) | {"media"}


shutil.copytree(
    root / "nomicous" / "backend",
    dest / "nomicous" / "backend",
    ignore=ignore_backend,
)
shutil.copytree(
    root / "nomicous" / "infrastructure",
    dest / "nomicous" / "infrastructure",
    ignore=ignore_deploy_artifacts,
)
shutil.copy2(root / "inference" / "__init__.py", dest / "inference" / "__init__.py")
shutil.copy2(root / "inference" / "admission.py", dest / "inference" / "admission.py")
shutil.copy2(root / "inference" / "registry.yaml", dest / "inference" / "registry.yaml")
shutil.copytree(
    root / "inference" / "contracts",
    dest / "inference" / "contracts",
    ignore=ignore_deploy_artifacts,
)
infrastructure_dest = dest / "inference" / "infrastructure"
infrastructure_dest.mkdir()
shutil.copy2(
    root / "inference" / "infrastructure" / "__init__.py",
    infrastructure_dest / "__init__.py",
)
shutil.copy2(
    root / "inference" / "infrastructure" / "settings.py",
    infrastructure_dest / "settings.py",
)
shutil.copytree(
    root / "inference" / "registry",
    dest / "inference" / "registry",
    ignore=ignore_deploy_artifacts,
)
shutil.copy2(root / "nomicous" / "VERSION", dest / "nomicous" / "VERSION")
PY
