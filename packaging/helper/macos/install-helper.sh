#!/usr/bin/env bash
# Install Nomicous Inference Helper from the mounted .dmg payload.
set -euo pipefail

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required to verify the helper after installation." >&2
  exit 1
fi

APP_SRC="$(cd "$(dirname "$0")" && pwd)/Applications/Nomicous Inference Helper.app"
APP_DST="$HOME/Applications/Nomicous Inference Helper.app"
APP_STAGE="$HOME/Applications/.Nomicous Inference Helper.staging.app"
APP_BACKUP="$HOME/Applications/.Nomicous Inference Helper.previous.app"
INSTALL_DIR="$APP_DST/Contents/MacOS"

if [ ! -d "$APP_SRC" ]; then
  echo "ERROR: helper app not found at '$APP_SRC'." >&2
  exit 1
fi

mkdir -p "$HOME/Applications" "$HOME/.nomicous/logs" "$HOME/.nomicous/hf/cache"
rm -rf "$APP_STAGE" "$APP_BACKUP"
cp -R "$APP_SRC" "$APP_STAGE"

PLIST_DST="$HOME/Library/LaunchAgents/com.nomicous.inference-helper.plist"
PLIST_BACKUP="${PLIST_DST}.previous"
mkdir -p "$(dirname "$PLIST_DST")"

PREVIOUS_LOADED=false
BACKUP_CREATED=false
SWAP_COMPLETED=false
PLIST_CHANGED=false
INSTALL_SUCCEEDED=false
if launchctl print "gui/$(id -u)/com.nomicous.inference-helper" >/dev/null 2>&1; then
  PREVIOUS_LOADED=true
fi

restore_previous_install() {
  status=$?
  if [ "$INSTALL_SUCCEEDED" != true ]; then
    set +e
    launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null
    if [ "$SWAP_COMPLETED" = true ]; then
      rm -rf "$APP_DST"
    fi
    if [ "$PLIST_CHANGED" = true ]; then
      if [ -f "$PLIST_BACKUP" ]; then
        mv "$PLIST_BACKUP" "$PLIST_DST"
      else
        rm -f "$PLIST_DST"
      fi
    fi
    if [ "$BACKUP_CREATED" = true ] && [ -d "$APP_BACKUP" ]; then
      mv "$APP_BACKUP" "$APP_DST"
    fi
    if [ "$PREVIOUS_LOADED" = true ] && [ -d "$APP_DST" ] && [ -f "$PLIST_DST" ]; then
      launchctl bootstrap "gui/$(id -u)" "$PLIST_DST" 2>/dev/null
    fi
  fi
  rm -rf "$APP_STAGE"
  exit "$status"
}
trap restore_previous_install EXIT

rm -f "$PLIST_BACKUP"
if [ -f "$PLIST_DST" ]; then
  cp "$PLIST_DST" "$PLIST_BACKUP"
fi
launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true

if [ -d "$APP_DST" ]; then
  mv "$APP_DST" "$APP_BACKUP"
  BACKUP_CREATED=true
fi
mv "$APP_STAGE" "$APP_DST"
SWAP_COMPLETED=true

PLIST_CHANGED=true
sed \
  -e "s|__INSTALL_DIR__|$INSTALL_DIR|g" \
  -e "s|__HOME__|$HOME|g" \
  -e "s|__REGISTRY_URL__|${HELPER_REGISTRY_URL:-https://api.nomicous.com/inference/v1/registry}|g" \
  "$INSTALL_DIR/../Resources/com.nomicous.inference-helper.plist" > "$PLIST_DST"

launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"

for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:8001/health >/dev/null; then
    INSTALL_SUCCEEDED=true
    rm -rf "$APP_BACKUP"
    rm -f "$PLIST_BACKUP"
    echo "Installed Nomicous Inference Helper."
    exit 0
  fi
  sleep 1
done

echo "ERROR: helper did not become ready. See $HOME/.nomicous/logs/inference-helper.log." >&2
exit 1
