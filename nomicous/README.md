# nomicous

AI-assisted transcription platform for manuscript pages. `nomicous/` is the production app root: the Postgres-backed backend, Vite frontend, and migration infrastructure live here, while the repository-level `model/` workspace remains outside nomicous.

See `CONTEXT.md` for domain glossary and `issues/prd.md` for requirements.

## Data layout

All persistence is under `data/` (path configurable via `NOMICOUS_DATA_ROOT` in `backend/.env`):

| Path | Contents |
|------|----------|
| `data/manuscripts/pages/` | Source page images (`.jpg`, `.png`, …) |
| `data/transcriptions/pages/` | Page transcriptions (`<stem>.txt`, line-broken) |
| `data/annotations/pages/` | Per-page annotation JSON (segments + pairings; tool-internal) |
| `data/manuscripts/export/` | **Exported outputs** — paired `<stem>_<segment_number>.jpg` and `.txt` side by side |

Missing subdirectories are **created automatically** when the API starts. If creation fails (permissions, bad path), startup aborts with a clear error pointing at `NOMICOUS_DATA_ROOT`.

### Sample fixture

`data/manuscripts/pages/sample_folio.jpg` and `data/transcriptions/pages/sample_folio.txt` are included for local testing.

## Setup

### Backend environment

```bash
cd nomicous
uv sync --project .. --group platform
cp backend/core/.env.example backend/core/.env
```

### Frontend

```bash
cd nomicous/frontend
cp .env.local.example .env.local   # VITE_API_BASE_URL defaults to http://localhost:8000
npm install
```

## Development (two terminals)

### Terminal 1 — API

```bash
docker compose up db -d
cd nomicous
uv run --project .. --group platform alembic -c infrastructure/alembic.ini upgrade head
uv run --project .. --group platform uvicorn backend.core.app:create_app --factory --reload
```

Default API URL: `http://127.0.0.1:8000`

### Terminal 2 — Frontend

```bash
cd nomicous/frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Docker (one command)

Ensure ports **5173**, **8000**, **8001**, and **5433** are free. Run Compose from the repository root:

```bash
cd ..
docker compose up --build      # foreground; Ctrl+C stops the stack
docker compose up --build -d   # detached background
```

| Service | URL |
|---------|-----|
| Frontend | [http://localhost:5173](http://localhost:5173) |
| API | [http://localhost:8000](http://localhost:8000) |
| ML inference API | [http://localhost:8001](http://localhost:8001) (separate service; health only today) |
| Postgres | `localhost:5433` |

Platform media is mounted at `nomicous/backend/media/`. The existing `data/` folder is not mounted or migrated by the production platform relocation.

Compose also starts **`inference-api`** and **`inference-worker`** from the repository-level [`inference/`](../inference/) package. That separate inference service owns contracts, registry-backed sync runs, and async job submission. `inference-api` is the HTTP boundary; `inference-worker` is reserved for slow CPU/GPU inference work. The platform API calls it through `INFERENCE_URL`. See [`inference/README.md`](../inference/README.md).

Useful after `-d`: `docker compose ps`, `docker compose logs -f`, `docker compose down`.

- `VITE_API_BASE_URL` (build/runtime env) — URL the browser uses (`http://localhost:8000`)

Rebuild the frontend image if you change `VITE_API_BASE_URL`.

### Bumping the Docker version

**Single source of truth:** [`VERSION`](VERSION) at the repo root of `nomicous/`.

1. Edit `VERSION` (semver, one line — e.g. `0.2.1`).
2. From the repository root, export it and rebuild so Compose tags images correctly:

```bash
cd ..
export NOMICOUS_VERSION=$(cat nomicous/VERSION)
docker compose up --build -d
```

The root `docker-compose.yml` builds `nomicous-api:${NOMICOUS_VERSION}` and `nomicous-frontend:${NOMICOUS_VERSION}` and passes `APP_VERSION` into both Dockerfiles (API `/health` and frontend build).

3. Verify:

```bash
curl -s http://localhost:8000/health | python -m json.tool
docker images 'nomicous-*'
```

**When to bump:** any release you want to distinguish in image tags or `/health` — not required for every code change during dev (rebuild with the same `NOMICOUS_VERSION` is fine).

**Optional:** set `NOMICOUS_VERSION` in a shell profile or root `.env` next to `docker-compose.yml` so you do not export it every time. If `NOMICOUS_VERSION` is unset, Compose defaults to `0.3.3` — keep it in sync with `VERSION` when tagging releases.

## Environment variables

| Variable | File | Default | Purpose |
|----------|------|---------|---------|
| `DATABASE_URL` | `backend/core/.env` | `postgresql+asyncpg://postgres:dev@localhost:5433/kalamos` | Async platform database URL |
| `SYNC_DATABASE_URL` | `backend/core/.env` | `postgresql://postgres:dev@localhost:5433/kalamos` | Alembic database URL |
| `JWT_SECRET` | `backend/core/.env` | development secret | Auth token signing key |
| `CORS_ORIGINS` | `backend/core/.env` | `http://localhost:3000,http://localhost:5173` | Allowed browser origins |
| `MEDIA_ROOT` | `backend/core/.env` | `nomicous/backend/media` | Uploaded document part media |
| `DEFAULT_SEGMENT_MODEL` | `backend/core/.env` | `kraken-segment-default` | Dev catalog name/ID for default segmentation |
| `DEFAULT_TRANSCRIBE_MODEL` | `backend/core/.env` | `kraken-transcribe-default` | Dev catalog name/ID for default transcription |
| `KRAKEN_MODEL_PATH` | `backend/core/.env` | `../model/kraken` | Local directory containing Kraken weights |
| `VITE_API_BASE_URL` | `frontend/.env.local` | `http://localhost:8000` | Frontend → API URL |

## Inference Catalog

Keep Kraken weights in the repository-level `model/` workspace, not under
`nomicous/`. The dev seed records `KRAKEN_MODEL_PATH/segment.mlmodel` and
`KRAKEN_MODEL_PATH/transcribe.mlmodel` as artifact references and creates
project-level defaults:

```bash
cd nomicous
alembic -c infrastructure/alembic.ini upgrade head
python ../scripts/platform/seed_dev_inference.py
```

## Tests (TDD)

```bash
# From repository root
uv run --group platform pytest tests/nomicous/unit
uv run --group platform pytest tests/nomicous/integration -m "not ml"
```

See [`docs/testing.md`](../docs/testing.md) for the ML lane and full-suite commands.

## OpenAPI

```bash
python scripts/platform/export_openapi.py

cd nomicous/frontend
npm run codegen:api
```

Generated schema types: `frontend/src/types/openapi.ts`. App-facing aliases: `frontend/src/types/api.ts`.

