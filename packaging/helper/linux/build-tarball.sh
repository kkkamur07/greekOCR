#!/usr/bin/env bash
# Build a Linux tarball with systemd user unit for the Inference Helper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HELPER_DIR="$ROOT/packaging/helper"
DIST_DIR="$HELPER_DIR/dist"
INSTALL_DIR="$DIST_DIR/linux-installer"
ARCHIVE="$DIST_DIR/nomicous-inference-helper-linux.tar.gz"

"$HELPER_DIR/scripts/build-pyinstaller.sh"

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/nomicous-inference-helper"
cp -R "$DIST_DIR/nomicous-inference-helper/"* "$INSTALL_DIR/nomicous-inference-helper/"
cp "$SCRIPT_DIR/nomicous-inference-helper.service" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/install-helper.sh" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/diagnose-helper.sh" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/install-helper.sh"
chmod +x "$INSTALL_DIR/diagnose-helper.sh"

tar -czf "$ARCHIVE" -C "$INSTALL_DIR" .
echo "Built $ARCHIVE"
