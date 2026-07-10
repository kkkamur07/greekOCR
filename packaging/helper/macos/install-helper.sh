#!/usr/bin/env bash
# Install Nomicous Inference Helper from the mounted .dmg payload.
set -euo pipefail

APP_SRC="$(cd "$(dirname "$0")" && pwd)/Applications/Nomicous Inference Helper.app"
APP_DST="/Applications/Nomicous Inference Helper.app"
INSTALL_DIR="$APP_DST/Contents/MacOS"

mkdir -p "$HOME/.nomicous/logs" "$HOME/.nomicous/hf/cache"
cp -R "$APP_SRC" /Applications/

PLIST_DST="$HOME/Library/LaunchAgents/com.nomicous.inference-helper.plist"
sed \
  -e "s|__INSTALL_DIR__|$INSTALL_DIR/nomicous-inference-helper|g" \
  -e "s|__HOME__|$HOME|g" \
  -e "s|__REGISTRY_URL__|${HELPER_REGISTRY_URL:-https://api.nomicous.com/inference/v1/registry}|g" \
  -e "s|__CORS_ORIGINS__|${HELPER_CORS_ORIGINS:-https://app.nomicous.com}|g" \
  "$INSTALL_DIR/../Resources/com.nomicous.inference-helper.plist" > "$PLIST_DST"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed Nomicous Inference Helper. Probe http://127.0.0.1:8001/health"
