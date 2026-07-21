#!/usr/bin/env bash
# Diagnose a Linux helper that is not reachable on loopback.
set -u

URL="http://127.0.0.1:8001/health"
LOG_FILE="$HOME/.nomicous/logs/inference-helper.log"
SERVICE="nomicous-inference-helper.service"

if curl --fail --silent --show-error --max-time 3 "$URL" >/dev/null; then
  echo "Inference Helper is ready at $URL"
  exit 0
fi

echo "Inference Helper is not reachable at $URL"
echo

if command -v systemctl >/dev/null 2>&1 \
  && systemctl --user show-environment >/dev/null 2>&1; then
  echo "systemd user service:"
  systemctl --user status "$SERVICE" --no-pager || true
  echo
  echo "Recent service logs:"
  journalctl --user -u "$SERVICE" -n 50 --no-pager || true
else
  echo "No usable systemd user session was found."
  echo "The installer should have used desktop autostart instead."
fi

if command -v ss >/dev/null 2>&1; then
  echo
  echo "Listening sockets:"
  ss -ltnp || true
fi

if [ -f "$LOG_FILE" ]; then
  echo
  echo "Helper log:"
  tail -n 50 "$LOG_FILE"
else
  echo
  echo "No helper log found at $LOG_FILE"
fi

exit 1
