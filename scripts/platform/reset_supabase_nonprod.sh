#!/usr/bin/env bash
# Reset the application schema on a disposable, non-production Supabase project.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${SUPABASE_ENV_FILE:-$ROOT/nomicous/backend/core/.env.supabase}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

set -a
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  export "$line"
done < "$ENV_FILE"
set +a

if [[ "${SUPABASE_NON_PRODUCTION:-}" != "true" ]]; then
  echo "Set SUPABASE_NON_PRODUCTION=true in the environment file." >&2
  exit 1
fi
if [[ "${CONFIRM_SUPABASE_RESET:-}" != "RESET" ]]; then
  echo "Set CONFIRM_SUPABASE_RESET=RESET to confirm this destructive reset." >&2
  exit 1
fi
if [[ -z "${MIGRATOR_DATABASE_URL:-}" ]]; then
  echo "MIGRATOR_DATABASE_URL is required in $ENV_FILE" >&2
  exit 1
fi

psql "$MIGRATOR_DATABASE_URL" -v ON_ERROR_STOP=1 <<'SQL'
DROP TABLE IF EXISTS
  auth_sessions,
  auth_rate_limit_attempts,
  media_deletion_intents,
  annotation_history_snapshots,
  page_transcription_lines,
  line_transcriptions,
  transcriptions,
  inference_jobs,
  jobs,
  model_bindings,
  inference_models,
  lines,
  blocks,
  document_parts,
  documents,
  project_shared_users,
  projects,
  users,
  alembic_version
CASCADE;

DROP TYPE IF EXISTS
  inference_job_status,
  job_status,
  job_type,
  binding_task,
  transcription_kind,
  line_source,
  line_geometry_kind,
  document_workflow,
  inference_task
CASCADE;

DROP FUNCTION IF EXISTS app_user_can_access_binding(uuid, uuid, uuid);
DROP FUNCTION IF EXISTS app_user_can_access_part(uuid);
DROP FUNCTION IF EXISTS app_user_can_access_document(uuid);
DROP FUNCTION IF EXISTS app_user_can_access_project(uuid);
DROP FUNCTION IF EXISTS app_auth_lookup_enabled();
DROP FUNCTION IF EXISTS app_public_read_enabled();
DROP FUNCTION IF EXISTS app_current_user_id();
DROP FUNCTION IF EXISTS app_rls_bypass();
SQL

cd "$ROOT"
./scripts/platform/migrate_supabase.sh
uv run python scripts/platform/seed_dev_nomicous.py

echo "Supabase schema reset and migrations applied."
echo "Storage objects are not deleted; clear the disposable bucket separately if needed."
