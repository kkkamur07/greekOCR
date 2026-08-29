#!/usr/bin/env bash
# Bundle platform API sources into infrastructure/platform for Vercel.
set -euo pipefail

ROOT="${PLATFORM_BUNDLE_SOURCE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
DEST="${PLATFORM_BUNDLE_DEST:-$(cd "$(dirname "$0")" && pwd)}"

if [[ "$DEST" == "/" ]]; then
    echo "Refusing to use the filesystem root as a platform bundle destination." >&2
    exit 1
fi

rm -rf "$DEST/nomikos" "$DEST/nomikos_inference"
mkdir -p "$DEST/nomikos" "$DEST/nomikos_inference"

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
    root / "nomikos" / "backend",
    dest / "nomikos" / "backend",
    ignore=ignore_backend,
)
shutil.copytree(
    root / "nomikos" / "infrastructure",
    dest / "nomikos" / "infrastructure",
    ignore=ignore_deploy_artifacts,
)
shutil.copy2(root / "nomikos_inference" / "__init__.py", dest / "nomikos_inference" / "__init__.py")
shutil.copy2(root / "nomikos_inference" / "admission.py", dest / "nomikos_inference" / "admission.py")
shutil.copy2(root / "nomikos_inference" / "settings.py", dest / "nomikos_inference" / "settings.py")
shutil.copy2(root / "nomikos_inference" / "registry.yaml", dest / "nomikos_inference" / "registry.yaml")
shutil.copytree(
    root / "nomikos_inference" / "contracts",
    dest / "nomikos_inference" / "contracts",
    ignore=ignore_deploy_artifacts,
)
shutil.copytree(
    root / "nomikos_inference" / "registry",
    dest / "nomikos_inference" / "registry",
    ignore=ignore_deploy_artifacts,
)
shutil.copy2(root / "nomikos" / "VERSION", dest / "nomikos" / "VERSION")
PY
