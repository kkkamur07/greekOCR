#!/usr/bin/env bash
# Shared PyInstaller build for all Inference Helper platform installers.
set -euo pipefail

HELPER_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$HELPER_DIR"
# Build tools live in the `packaging` dependency group so a single `uv run`
# keeps them in sync alongside the runtime `inference` group. Installing them
# separately (pip/uv pip) is unreliable: a later `uv run` re-syncs the venv to
# the lockfile and silently drops the externally-installed build tools.
uv run --group inference --group packaging pyinstaller --noconfirm pyinstaller.spec
