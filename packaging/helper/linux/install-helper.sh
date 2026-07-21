#!/usr/bin/env bash
# Install Nomicous Inference Helper from the extracted Linux tarball.
set -euo pipefail

INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nomicous/inference-helper"
STAGE_ROOT="${INSTALL_ROOT}.staging.$$"
BACKUP_ROOT="${INSTALL_ROOT}.previous"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGISTRY_URL="${HELPER_REGISTRY_URL:-https://api.nomicous.com/inference/v1/registry}"

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required to verify the helper after installation." >&2
  exit 1
fi

mkdir -p "$(dirname "$INSTALL_ROOT")" "$HOME/.nomicous/logs" "$HOME/.nomicous/hf/cache"
rm -rf "$STAGE_ROOT" "$BACKUP_ROOT"
mkdir -p "$STAGE_ROOT"
cp -R "$SCRIPT_DIR/nomicous-inference-helper/"* "$STAGE_ROOT/"
cp "$SCRIPT_DIR/diagnose-helper.sh" "$STAGE_ROOT/"
chmod +x "$STAGE_ROOT/diagnose-helper.sh"

UNIT_DST="$HOME/.config/systemd/user/nomicous-inference-helper.service"
UNIT_BACKUP="${UNIT_DST}.previous"
AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/nomicous-inference-helper.desktop"
AUTOSTART_BACKUP="${AUTOSTART_FILE}.previous"

USE_SYSTEMD=false
PREVIOUS_SYSTEMD_ACTIVE=false
PREVIOUS_FALLBACK_ACTIVE=false
BACKUP_CREATED=false
SWAP_COMPLETED=false
CONFIG_CHANGED=false
INSTALL_SUCCEEDED=false
restore_previous_install() {
  status=$?
  if [ "$INSTALL_SUCCEEDED" != true ]; then
    set +e
    if [ "$USE_SYSTEMD" = true ]; then
      systemctl --user stop nomicous-inference-helper.service
    elif command -v pkill >/dev/null 2>&1; then
      pkill -f -- "$INSTALL_ROOT/nomicous-inference-helper"
    fi
    if [ "$SWAP_COMPLETED" = true ]; then
      rm -rf "$INSTALL_ROOT"
    fi
    if [ "$BACKUP_CREATED" = true ] && [ -d "$BACKUP_ROOT" ]; then
      mv "$BACKUP_ROOT" "$INSTALL_ROOT"
    fi
    if [ "$CONFIG_CHANGED" = true ]; then
      if [ -f "$UNIT_BACKUP" ]; then
        mv "$UNIT_BACKUP" "$UNIT_DST"
      else
        rm -f "$UNIT_DST"
      fi
      if [ -f "$AUTOSTART_BACKUP" ]; then
        mkdir -p "$AUTOSTART_DIR"
        mv "$AUTOSTART_BACKUP" "$AUTOSTART_FILE"
      else
        rm -f "$AUTOSTART_FILE"
      fi
    fi
    if [ "$PREVIOUS_FALLBACK_ACTIVE" = true ] && [ -d "$INSTALL_ROOT" ]; then
      if [ -x "$INSTALL_ROOT/run-helper.sh" ]; then
        nohup "$INSTALL_ROOT/run-helper.sh" \
          >>"$HOME/.nomicous/logs/inference-helper.log" 2>&1 &
      else
        HELPER_REGISTRY_URL="$REGISTRY_URL" nohup "$INSTALL_ROOT/nomicous-inference-helper" \
          >>"$HOME/.nomicous/logs/inference-helper.log" 2>&1 &
      fi
    elif [ "$PREVIOUS_SYSTEMD_ACTIVE" = true ]; then
      systemctl --user daemon-reload
      systemctl --user start nomicous-inference-helper.service
    fi
  fi
  rm -rf "$STAGE_ROOT"
  exit "$status"
}
trap restore_previous_install EXIT

if command -v systemctl >/dev/null 2>&1 \
  && systemctl --user show-environment >/dev/null 2>&1; then
  USE_SYSTEMD=true
  if systemctl --user is-active --quiet nomicous-inference-helper.service; then
    PREVIOUS_SYSTEMD_ACTIVE=true
  fi
  systemctl --user stop nomicous-inference-helper.service 2>/dev/null || true
fi
if command -v pgrep >/dev/null 2>&1 \
  && pgrep -f -- "$INSTALL_ROOT/nomicous-inference-helper" >/dev/null 2>&1; then
  PREVIOUS_FALLBACK_ACTIVE=true
  pkill -f -- "$INSTALL_ROOT/nomicous-inference-helper" 2>/dev/null || true
fi

if [ -d "$INSTALL_ROOT" ]; then
  mv "$INSTALL_ROOT" "$BACKUP_ROOT"
  BACKUP_CREATED=true
fi
mv "$STAGE_ROOT" "$INSTALL_ROOT"
SWAP_COMPLETED=true

mkdir -p "$(dirname "$UNIT_DST")" "$AUTOSTART_DIR"
rm -f "$UNIT_BACKUP" "$AUTOSTART_BACKUP"
if [ -f "$UNIT_DST" ]; then
  cp "$UNIT_DST" "$UNIT_BACKUP"
fi
if [ -f "$AUTOSTART_FILE" ]; then
  cp "$AUTOSTART_FILE" "$AUTOSTART_BACKUP"
fi
CONFIG_CHANGED=true
sed \
  -e "s|__INSTALL_DIR__|$INSTALL_ROOT|g" \
  -e "s|__HOME__|$HOME|g" \
  -e "s|__REGISTRY_URL__|$REGISTRY_URL|g" \
  "$SCRIPT_DIR/nomicous-inference-helper.service" > "$UNIT_DST"

if [ "$USE_SYSTEMD" = true ]; then
  systemctl --user daemon-reload
  systemctl --user enable --now nomicous-inference-helper.service
  rm -f "$AUTOSTART_FILE"
  USE_SYSTEMD=true
else
  mkdir -p "$AUTOSTART_DIR"
  RUNNER="$INSTALL_ROOT/run-helper.sh"
  cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
export HELPER_REGISTRY_URL='$REGISTRY_URL'
'$INSTALL_ROOT/nomicous-inference-helper' &
HELPER_PID=\$!
echo "\$HELPER_PID" > '$INSTALL_ROOT/helper.pid'
cleanup() {
  kill "\$HELPER_PID" 2>/dev/null || true
  rm -f '$INSTALL_ROOT/helper.pid'
}
trap cleanup EXIT INT TERM
wait "\$HELPER_PID"
EOF
  chmod +x "$RUNNER"
  cat > "$AUTOSTART_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Nomicous Inference Helper
Exec="$RUNNER"
NoDisplay=true
X-GNOME-Autostart-enabled=true
EOF
  nohup "$RUNNER" \
    >>"$HOME/.nomicous/logs/inference-helper.log" 2>&1 &
fi

for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 2 http://127.0.0.1:8001/health >/dev/null; then
    INSTALL_SUCCEEDED=true
    rm -rf "$BACKUP_ROOT"
    rm -f "$UNIT_BACKUP" "$AUTOSTART_BACKUP"
    if [ "$USE_SYSTEMD" = true ]; then
      echo "Installed Nomicous Inference Helper with systemd."
    else
      echo "Installed Nomicous Inference Helper with desktop autostart."
    fi
    exit 0
  fi
  sleep 1
done

echo "ERROR: helper did not become ready. See $HOME/.nomicous/logs/inference-helper.log." >&2
exit 1
