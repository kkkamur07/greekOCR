#!/usr/bin/env bash
# Verify repository-controlled hardening without production credentials.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

uv sync --all-groups
npm --prefix nomicous/frontend ci

npm --prefix nomicous/frontend run typecheck
npm --prefix nomicous/frontend run lint
npm --prefix nomicous/frontend test
npm --prefix nomicous/frontend run build
npm --prefix nomicous/frontend run check:api

uv run --locked --group dev poe test-fast
uv run --locked --group inference pytest tests/inference tests/hf -m "not integration"
uv run --locked --group dev pre-commit run frontend-eslint --files \
  nomicous/frontend/src/api/client.ts
uv run --locked --group dev pre-commit run frontend-prettier --files \
  nomicous/frontend/package.json
git diff --check

if [[ "${VERIFY_INTEGRATION:-0}" == "1" ]]; then
  uv run --locked --group platform --group inference \
    pytest tests/nomicous/integration -m "not ml"
fi

if [[ "${VERIFY_DOCKER:-0}" == "1" ]]; then
  bash deploy/platform/build.sh
  docker build --target runtime --tag nomicous-api:verify -f nomicous/Dockerfile .
  docker build --target runtime --tag nomicous-inference:verify -f inference/Dockerfile .
  docker run --rm --entrypoint python nomicous-api:verify \
    -c "from backend.core.main import app; assert app.title"
  docker run --rm --entrypoint python nomicous-inference:verify \
    -c "from inference.api.main import app; assert app.title"
fi
