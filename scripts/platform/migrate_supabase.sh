#!/usr/bin/env bash
# Apply Alembic migrations to a Supabase Postgres database.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${SUPABASE_ENV_FILE:-$ROOT/nomicous/backend/core/.env.supabase}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  echo "Copy nomicous/backend/core/.env.supabase.example and fill in credentials." >&2
  exit 1
fi

set -a
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  export "$line"
done < "$ENV_FILE"
set +a

if [[ -z "${MIGRATOR_DATABASE_URL:-}" ]]; then
  echo "MIGRATOR_DATABASE_URL is required in $ENV_FILE" >&2
  exit 1
fi

cd "$ROOT/nomicous"
echo "Running Alembic against Supabase (migrator URL)…"
PYTHONPATH=. alembic -c infrastructure/alembic.ini upgrade head
echo "Done. Current revision:"
PYTHONPATH=. alembic -c infrastructure/alembic.ini current
