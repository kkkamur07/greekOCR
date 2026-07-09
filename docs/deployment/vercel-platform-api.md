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

Do not set a custom install command. Vercel's Python runtime installs from the
checked-in `deploy/platform/requirements.txt`.

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
- `inference/contracts/`
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
BEHIND_PROXY=true
FORWARDED_ALLOW_IPS=*
STORAGE_BACKEND=supabase
ENABLE_TEST_JOB_ROUTES=false
```

Run the platform worker and inference workers on a persistent host if cloud
inference is enabled.

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
