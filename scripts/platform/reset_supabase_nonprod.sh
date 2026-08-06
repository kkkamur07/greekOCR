#!/usr/bin/env bash
# Reset the application schema on a disposable, non-production Supabase project.
#
# Every guard below reads the env file directly instead of the process
# environment. The previous version checked $SUPABASE_NON_PRODUCTION *after*
# merging the file into the environment, so an `export SUPABASE_NON_PRODUCTION=true`
# sitting in a shell profile satisfied a guard whose error message claimed to be
# talking about the file - including when SUPABASE_ENV_FILE had been pointed at
# production credentials. Provenance is the whole point of these checks, so the
# file is parsed before anything is merged.
#
# Required *in the env file* (exporting them in your shell does nothing):
#   SUPABASE_NON_PRODUCTION=true
#   ENVIRONMENT=<anything but production>
#   MIGRATOR_DATABASE_URL=<the database whose schema gets dropped>
#
# Confirmation of the resolved target (project ref, or host/db for non-Supabase):
#   interactive: type the target when prompted
#   unattended:  CONFIRM_SUPABASE_RESET=<target> plus --yes
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${SUPABASE_ENV_FILE:-$ROOT/nomicous/backend/core/.env.supabase}"

# Captured before the merge so an env file cannot confirm its own destruction.
CONFIRM_INPUT="${CONFIRM_SUPABASE_RESET:-}"
PRODUCTION_REFS="${SUPABASE_PRODUCTION_PROJECT_REFS:-}"
ASSUME_YES="${SUPABASE_RESET_ASSUME_YES:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y | --yes)
      ASSUME_YES=true
      ;;
    -h | --help)
      echo "usage: $(basename "$0") [--yes]"
      echo
      echo "Drops the application schema described by SUPABASE_ENV_FILE"
      echo "(default: $ENV_FILE) and reapplies migrations."
      echo "--yes skips the interactive prompt and requires CONFIRM_SUPABASE_RESET"
      echo "to equal the target project ref."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1 (try --help)" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  echo "Copy nomicous/backend/core/.env.supabase.example and fill in credentials." >&2
  exit 1
fi

# Read one assignment straight out of the env file. Guards MUST use this rather
# than $VAR, which cannot distinguish a value the file set from one the caller
# exported. Last assignment wins, matching the merge loop below.
env_file_value() {
  local key="$1"
  local line trimmed value=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    trimmed="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$trimmed" || "$trimmed" == '#'* ]] && continue
    if [[ "$trimmed" == export[[:space:]]* ]]; then
      trimmed="${trimmed#export}"
      trimmed="${trimmed#"${trimmed%%[![:space:]]*}"}"
    fi
    [[ "$trimmed" == "$key="* ]] || continue
    value="${trimmed#"$key="}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ ${#value} -ge 2 && "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
      value="${value:1:${#value} - 2}"
    elif [[ ${#value} -ge 2 && "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value} - 2}"
    fi
  done <"$ENV_FILE"
  printf '%s' "$value"
}

# Supabase exposes the project ref three ways: https://<ref>.supabase.co, the
# direct database host db.<ref>.supabase.co, and the pooler username
# postgres.<ref>@...pooler.supabase.com. The trailing [:/] anchor keeps
# "pooler.supabase.com" from being read as the ref "pooler".
supabase_project_ref() {
  local url="$1"
  if [[ "$url" =~ (^|[/@.])([A-Za-z0-9-]+)\.supabase\.co([:/?]|$) ]]; then
    printf '%s' "${BASH_REMATCH[2]}"
    return 0
  fi
  if [[ "$url" =~ ://[^:/@]+\.([A-Za-z0-9]+)(:[^@]*)?@ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  fi
}

# What the operator has to recognise before we drop it. Falls back to host/db so
# a local or self-hosted Postgres still gets a specific, typed confirmation.
database_target() {
  local url="$1" ref host
  ref="$(supabase_project_ref "$url")"
  if [[ -n "$ref" ]]; then
    printf '%s' "$ref"
    return 0
  fi
  host="${url#*://}"
  host="${host#*@}"
  printf '%s' "${host%%\?*}"
}

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

FILE_NON_PRODUCTION="$(lowercase "$(env_file_value SUPABASE_NON_PRODUCTION)")"
FILE_ENVIRONMENT="$(lowercase "$(env_file_value ENVIRONMENT)")"
FILE_MIGRATOR_URL="$(env_file_value MIGRATOR_DATABASE_URL)"
FILE_SUPABASE_URL="$(env_file_value SUPABASE_URL)"

if [[ "$FILE_NON_PRODUCTION" != "true" ]]; then
  echo "Guard failed: SUPABASE_NON_PRODUCTION is not 'true' in $ENV_FILE" >&2
  echo "Add SUPABASE_NON_PRODUCTION=true to that file. An exported shell" >&2
  echo "variable is deliberately ignored - the flag must describe the file." >&2
  exit 1
fi

if [[ -z "$FILE_ENVIRONMENT" ]]; then
  echo "Guard failed: ENVIRONMENT is not set in $ENV_FILE" >&2
  echo "The reset cannot confirm the target is non-production without it." >&2
  exit 1
fi
if [[ "$FILE_ENVIRONMENT" == "production" ]]; then
  echo "Guard failed: $ENV_FILE declares ENVIRONMENT=production" >&2
  echo "Refusing to drop the schema of a production deployment." >&2
  exit 1
fi

if [[ -z "$FILE_MIGRATOR_URL" ]]; then
  echo "Guard failed: MIGRATOR_DATABASE_URL is not set in $ENV_FILE" >&2
  exit 1
fi

DB_REF="$(supabase_project_ref "$FILE_MIGRATOR_URL")"
API_REF="$(supabase_project_ref "$FILE_SUPABASE_URL")"
# A mismatch means the file mixes two projects - the classic way a production
# database URL ends up pasted into an otherwise harmless staging profile.
if [[ -n "$DB_REF" && -n "$API_REF" && "$DB_REF" != "$API_REF" ]]; then
  echo "Guard failed: $ENV_FILE mixes Supabase projects" >&2
  echo "  MIGRATOR_DATABASE_URL project ref: $DB_REF" >&2
  echo "  SUPABASE_URL project ref:          $API_REF" >&2
  exit 1
fi

TARGET="$(database_target "$FILE_MIGRATOR_URL")"

# Optional operator/CI denylist for refs that must never be reset, whatever a
# checked-in env file claims about itself.
for ref in ${PRODUCTION_REFS//,/ }; do
  if [[ -n "$ref" && ("$ref" == "$DB_REF" || "$ref" == "$TARGET") ]]; then
    echo "Guard failed: target '$TARGET' is listed in SUPABASE_PRODUCTION_PROJECT_REFS" >&2
    exit 1
  fi
done

echo "About to DROP the application schema:"
echo "  env file    : $ENV_FILE"
echo "  environment : $FILE_ENVIRONMENT"
echo "  target      : $TARGET"

if [[ "$ASSUME_YES" != "true" && -t 0 ]]; then
  read -r -p "Type the target ($TARGET) to confirm: " CONFIRM_INPUT
fi

if [[ "$CONFIRM_INPUT" != "$TARGET" ]]; then
  echo "Guard failed: confirmation did not match the target '$TARGET'" >&2
  echo "Interactively, type the target when prompted. Unattended, set" >&2
  echo "CONFIRM_SUPABASE_RESET=$TARGET and pass --yes." >&2
  exit 1
fi

set -a
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  # shellcheck disable=SC2163 # $line is a KEY=VALUE pair, not a variable name.
  export "$line"
done <"$ENV_FILE"
set +a

# Use the parsed value, not the merged environment: the merge only overrides
# ambient state for keys the file actually declares.
# Every table the chain creates has to be named here. CASCADE drops dependent FK
# *constraints*, not dependent tables - so leaving helper_pairings/helper_devices
# off this list left them standing with `inference_host` intact while
# alembic_version went, and the replay then failed with "column already exists".
# A second reset is the case that finds it; the first one always worked. The
# squashed chain has no guarded CREATE TABLE left to paper over a partial drop,
# so this list is now the only thing making a reset repeatable.
psql "$FILE_MIGRATOR_URL" -v ON_ERROR_STOP=1 <<'SQL'
DROP TABLE IF EXISTS
  auth_sessions,
  auth_rate_limit_attempts,
  media_deletion_intents,
  annotation_history_snapshots,
  page_transcription_lines,
  line_transcriptions,
  transcriptions,
  inference_jobs,  -- pre-006 databases only; harmless once dropped
  jobs,
  model_bindings,
  inference_models,
  lines,
  blocks,
  document_parts,
  documents,
  project_shared_users,
  projects,
  helper_pairings,  -- holds the FK to helper_devices, so it goes first
  helper_devices,
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
  execution_target,
  inference_task
CASCADE;

-- Created by 001 alongside the trigger on jobs. Dropping the table takes the
-- trigger, but a function is schema-level and survives every table drop above.
DROP FUNCTION IF EXISTS jobs_execution_target_is_fixed();
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

echo "Supabase schema reset and migrations applied for $TARGET."
echo "Storage objects are not deleted; clear the disposable bucket separately if needed."
