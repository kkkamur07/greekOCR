#!/usr/bin/env bash
# Install Nomicous Inference Helper from the extracted Linux tarball.
set -euo pipefail

INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nomicous/inference-helper"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$INSTALL_ROOT" "$HOME/.nomicous/logs" "$HOME/.nomicous/hf/cache"
cp -R "$SCRIPT_DIR/nomicous-inference-helper/"* "$INSTALL_ROOT/"

UNIT_DST="$HOME/.config/systemd/user/nomicous-inference-helper.service"
mkdir -p "$(dirname "$UNIT_DST")"
sed \
  -e "s|__INSTALL_DIR__|$INSTALL_ROOT|g" \
  -e "s|__HOME__|$HOME|g" \
  -e "s|__REGISTRY_URL__|${HELPER_REGISTRY_URL:-https://api.nomicous.example/inference/v1/registry}|g" \
  "$SCRIPT_DIR/nomicous-inference-helper.service" > "$UNIT_DST"

systemctl --user daemon-reload
systemctl --user enable --now nomicous-inference-helper.service

echo "Installed Nomicous Inference Helper. Probe http://127.0.0.1:8001/health"
