#!/usr/bin/env bash
# Export platform-prod deps and install them for Vercel (PEP 668 / uv-managed Python).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$(cd "$(dirname "$0")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

cd "$ROOT"
uv export --only-group platform-prod --no-hashes -o "$DEST/requirements.txt"
uv pip install -r "$DEST/requirements.txt" --python "$PYTHON_VERSION" --system
