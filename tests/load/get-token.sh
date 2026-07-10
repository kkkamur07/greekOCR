#!/usr/bin/env bash
#
# Load a dedicated test user's JWT into the current shell.
#
# Usage:
#   # Or put these values in tests/load/.env (ignored by git).
#   export LOCUST_EMAIL="load-test@example.com"
#   export LOCUST_PASSWORD="..."
#   source tests/load/get-token.sh

load_locust_token() {
  local script_dir source_path response token

  if [[ -n "${BASH_SOURCE:-}" ]]; then
    source_path="${BASH_SOURCE[0]}"
  elif [[ -n "${ZSH_VERSION:-}" ]]; then
    source_path="${(%):-%x}"
  else
    source_path="$0"
  fi
  script_dir="$(cd -- "$(dirname -- "${source_path}")" && pwd)"
  if [[ -f "${script_dir}/.env" ]]; then
    set -a
    source "${script_dir}/.env"
    set +a
  fi

  if [[ -z "${LOCUST_EMAIL:-}" ]]; then
    echo "Set LOCUST_EMAIL before sourcing this script" >&2
    return 1
  fi
  if [[ -z "${LOCUST_PASSWORD:-}" ]]; then
    echo "Set LOCUST_PASSWORD before sourcing this script" >&2
    return 1
  fi

  if ! response="$(
    curl --fail-with-body --silent --show-error \
      --connect-timeout 5 \
      --max-time 15 \
      -X POST "https://api.nomicous.com/auth/login" \
      -H "Content-Type: application/json" \
      --data "{\"email\":\"${LOCUST_EMAIL}\",\"password\":\"${LOCUST_PASSWORD}\"}"
  )"; then
    echo "API login failed; check the load-test credentials" >&2
    return 1
  fi

  if ! token="$(
    python3 -c '
import json
import sys

payload = json.load(sys.stdin)
token = payload.get("access_token")
if not token:
    raise SystemExit("Login response did not contain access_token")
print(token)
' <<<"${response}"
  )"; then
    echo "Login response did not contain access_token" >&2
    return 1
  fi

  export LOCUST_ACCESS_TOKEN="${token}"
  echo "LOCUST_ACCESS_TOKEN loaded."
}

load_locust_token
