# Vercel platform API deployment notes

This note records the deploy lessons from shipping **api.nomicous.com** from
`deploy/platform/`. It is intentionally practical: use it when a Vercel build
or runtime import fails.

## Project settings

| Setting | Value |
|---------|-------|
| Root Directory | `deploy/platform` |
| Framework | Other |
| Install Command | *(empty / default)* |
| Build Command | `bash build.sh` |
| Output Directory | `.` |
| Domain | `api.nomicous.com` |
| Function region | `fra1` (Frankfurt, Europe) |

Do not set a custom install command. Vercel's Python runtime installs from the
checked-in `deploy/platform/requirements.txt`.

The region applies to the platform API functions only. Static Vercel projects
remain globally edge-served. Compare API p95 latency with the Supabase project
region after deployment; change `regions` in `deploy/platform/vercel.json` if needed.

`build.sh` only bundles source files into the Vercel project root. It must not
install dependencies.

## Python version

`deploy/platform/.python-version` pins the function runtime to Python 3.12
because Vercel supports Python 3.12/3.13/3.14, while local development remains
Python 3.11 at the repository root.

Vercel logs can show two Python phases:

- An install resolver phase that may mention the current Vercel default Python.
- A function packaging phase that should say it is using Python from
  `deploy/platform/.python-version`.

The deploy should keep `requires-python = ">=3.11,<3.13"` so local 3.11 and
Vercel 3.12 are both valid.

## Build bundle shape

The generated copy under `deploy/platform/nomicous/` and
`deploy/platform/inference/` is a build artifact. It is gitignored and recreated
by `bash build.sh`.

`build.sh` uses Python `shutil` rather than `rsync` because the Vercel build
image does not include `rsync`.

The platform API needs only:

- `nomicous/backend/`
- `nomicous/infrastructure/`
- `nomicous/VERSION`
- `inference/registry.yaml`
- `inference/admission.py`
- `inference/contracts/`
- `inference/infrastructure/__init__.py`
- `inference/infrastructure/settings.py`
- `inference/registry/`

It does not bundle model weights, training code, inference runtimes, notebooks,
or local media.

## Runtime dependency rules

`deploy/platform/requirements.txt` is generated from the `platform-prod` group
in `pyproject.toml` and committed so Vercel can do its standard Python install.

Keep the Vercel API runtime small. It should include request/response API,
database, auth, storage, registry, and export dependencies only.

Required runtime dependencies include:

- FastAPI / Starlette / Pydantic
- SQLAlchemy, asyncpg, psycopg2-binary
- Supabase Storage client
- Pillow and ReportLab for media normalization and PDF/export features
- PyYAML because `backend.ml.api.registry` imports `inference.registry`, which
  parses `inference/registry.yaml`

Do not include these in `platform-prod` unless a Vercel API route imports them
directly:

- `opencv-python-headless`
- `numpy`
- `alembic`
- `uvicorn[standard]`
- `pypdf`
- Kraken, Calamari, Torch, TensorFlow, or model weights

Alembic stays in the local/dev `platform` group because migrations are run from
local or operational scripts, not inside the Vercel serverless function.

## Bundle size limit

Vercel Python functions have a 500 MB bundle limit. The first API bundle exceeded
that limit after `opencv-python-headless` pulled in NumPy and large binary
wheels.

The platform export crop/mask path previously used OpenCV/NumPy for one polygon
mask fill. It now uses Pillow so the Vercel API can avoid OpenCV and NumPy
entirely.

This change affects only approved line artifact export:

- `nomicous/backend/annotation/application/processing.py`
- `nomicous/backend/annotation/application/export_service.py`
- `POST /{document_id}/parts/{part_id}/export`

It does not affect local helper inference, cloud inference preprocessing,
Kraken segmentation, Calamari transcription, or training.

## Serverless settings

Vercel is request/response only for this API deployment.

Set:

```bash
JOB_WORKER_ENABLED=false
JOB_SSE_NOTIFICATIONS_ENABLED=false
CLOUD_INFERENCE_ENABLED=false
# Set true only when this is a fixed, allowlisted proxy source range.
BEHIND_PROXY=false
# FORWARDED_ALLOW_IPS=203.0.113.0/24
STORAGE_BACKEND=supabase
ENABLE_TEST_JOB_ROUTES=false
```

Run the platform worker and inference workers on a persistent host if cloud
inference is enabled.

Never use `FORWARDED_ALLOW_IPS=*`. The API accepts only explicit IP/CIDR
allowlists and uses `X-Forwarded-For` for rate limiting only when the direct
peer matches one. If Vercel cannot provide a stable proxy source range for this
deployment, leave `BEHIND_PROXY=false`.

## Common failures

| Symptom | Fix |
|---------|-----|
| `externally-managed-environment` from `pip` or `uv pip --system` | Remove custom install commands; let Vercel install from `requirements.txt` |
| `alembic requires Python>=3.10` while using Python 3.9 | Ensure `deploy/platform/.python-version` exists and uses supported Vercel Python |
| `rsync: command not found` | Keep `build.sh` on Python stdlib copy helpers |
| `No Output Directory named "public"` | Set output directory to `.` |
| Function bundle exceeds 500 MB | Remove heavy runtime deps from `platform-prod`; do not bundle inference stacks |
| `ModuleNotFoundError: No module named 'yaml'` | Keep `PyYAML` explicit in `platform-prod` |

## Verification

Before pushing deploy changes:

```bash
cd deploy/platform
PYENV_VERSION=3.11.10 bash build.sh
```

Inspect the generated bundle before deployment. It must contain the platform
source and registry contracts only; it must not contain `.env` files, local
media, model weights, training outputs, or private credentials. Treat a
bundle inspection failure as a release blocker.

From repo root, confirm the registry import still works:

```bash
PYTHONPATH=. uv run python - <<'PY'
from pathlib import Path
from inference.registry import load_registry
registry = load_registry(Path("inference/registry.yaml"))
print(f"registry ok: {len(registry.models)} models")
PY
```

After Vercel deploy:

```bash
curl -s https://api.nomicous.com/health
```

The first production smoke test must also cover authentication, a bounded
upload, job creation, job polling, registry access, storage access, and a
representative export. Confirm that expected client errors are sanitized and
that Vercel logs do not contain passwords, bearer tokens, submitted payloads,
filesystem paths, or infrastructure exceptions.

Record p50, p95, and p99 API latency, error rate, request volume, and job
completion health before the change and during the first hour after deployment.
Use representative European clients and compare the result with the Supabase
project region. Keep the prior deployment available while observing the
change.

Rollback when p95 latency is more than 50% above baseline, error rate exceeds
2× baseline, a critical flow breaks, or any security or cross-user isolation
issue appears. For a region-specific regression, remove `regions` from
`deploy/platform/vercel.json`, redeploy the last known-good version, and repeat
the health and smoke checks.

## July 2026 production incident and fixes

The first production deployment built successfully but returned
`FUNCTION_INVOCATION_FAILED` on every request. The failures were startup
validation errors, not a Vercel build or Frankfurt connectivity problem. They
were fixed in this order:

1. `FORWARDED_ALLOW_IPS=*` was rejected by the hardened settings. The unsafe
   value was removed.
2. `BEHIND_PROXY=true` was enabled without an explicit trusted proxy CIDR. It
   was changed to `false`; forwarded headers are not trusted until Vercel's
   source range is explicitly allowlisted.
3. Production required an HTTPS `INFERENCE_URL`. The optional cloud endpoint
   was set to `https://inference.nomicous.com`.
4. Production validation required non-placeholder
   `INFERENCE_WEBHOOK_SECRET` and `INFERENCE_SERVICE_SECRET`. Both were added
   to Vercel as encrypted Production variables from the local ignored secret
   store. Their values are never committed or printed.
5. The migrated Next.js app required `NEXT_PUBLIC_*` variables, while the
   Vercel project still contained the old `VITE_*` names. The public API,
   CSRF-cookie, test-job, and local-helper variables were corrected.

The API function is pinned to Frankfurt with `"regions": ["fra1"]` in
`deploy/platform/vercel.json`. The landing page and app remain globally served;
only the API function is region-pinned.

For the time being, local OCR uses the user's local helper:

```text
Browser on the user's machine → http://localhost:8001
```

This is configured as `NEXT_PUBLIC_INFERENCE_HELPER_URL` for the frontend and
is allowed by the frontend Content Security Policy. It must not replace the
API's production `INFERENCE_URL`: `localhost` inside a Vercel function refers
to that ephemeral function container, not the researcher's computer. Cloud
inference remains disabled until a persistent inference host is enabled.
