#!/usr/bin/env bash
# Shared PyInstaller build for all Inference Helper platform installers.
set -euo pipefail

HELPER_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$HELPER_DIR"
# Build in an isolated environment containing only the ONNX helper runtime and
# PyInstaller. This prevents an existing development venv from leaking Torch
# or native model modules into Analysis.
uv run --isolated --no-dev --group helper --group packaging \
  pyinstaller --noconfirm --clean pyinstaller.spec

BUNDLE_ROOT="$HELPER_DIR/dist/nomicous-inference-helper"
EXECUTABLE="$BUNDLE_ROOT/nomicous-inference-helper"
if [ "$(uname -s)" = "Darwin" ]; then
  BUNDLE_ROOT="$HELPER_DIR/dist/Nomicous Inference Helper.app"
  EXECUTABLE="$BUNDLE_ROOT/Contents/MacOS/nomicous-inference-helper"
fi

python "$HELPER_DIR/scripts/verify-bundle.py" "$BUNDLE_ROOT" "$EXECUTABLE"
