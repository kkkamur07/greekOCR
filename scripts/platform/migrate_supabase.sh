#!/usr/bin/env bash
# Apply Alembic migrations to a Supabase Postgres database.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${SUPABASE_ENV_FILE:-$ROOT/nomikos/backend/core/.env.supabase}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  echo "Copy nomikos/backend/core/.env.supabase.example and fill in credentials." >&2
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

# Resolve the project venv's alembic explicitly. A bare `alembic` picks up
# whichever shim is first on PATH - pyenv's usually is - and that interpreter
# has neither our revisions nor the project's SQLAlchemy, so the upgrade fails
# in a way that reads like a broken migration chain.
ALEMBIC="$ROOT/.venv/bin/alembic"
if [[ ! -x "$ALEMBIC" ]]; then
  ALEMBIC="$(command -v alembic || true)"
fi
if [[ -z "$ALEMBIC" ]]; then
  echo "No alembic found. Run 'uv sync' to create $ROOT/.venv." >&2
  exit 1
fi

cd "$ROOT/nomikos"
echo "Running Alembic against Supabase (migrator URL)…"
echo "  alembic: $ALEMBIC"
PYTHONPATH=. "$ALEMBIC" -c infrastructure/alembic.ini upgrade head
echo "Done. Current revision:"
PYTHONPATH=. "$ALEMBIC" -c infrastructure/alembic.ini current
