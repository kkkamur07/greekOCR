# nomicous

AI-assisted transcription platform for manuscript pages. `nomicous/` is the production app root: the Postgres-backed backend, Next.js frontend, and migration infrastructure live here, while the repository-level `model/` workspace remains outside nomicous.

See `CONTEXT.md` for domain glossary and [`docs/README.md`](../docs/README.md) for guides, deployment, and audits.

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

Place page images under `data/manuscripts/pages/` and matching line-broken
transcriptions under `data/transcriptions/pages/` for local testing. Uploaded
Document parts are stored as normal WebP objects (`parts/<uuid>/<stem>.webp`),
not special `folio*` names.

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
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL defaults to http://localhost:8000
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
| ML inference API | [http://localhost:8010](http://localhost:8010) (host port; container listens on 8001. Host 8001 is reserved for the local inference helper) |
| Postgres | `localhost:5433` |

Platform media is mounted at `nomicous/backend/media/`. The existing `data/` folder is not mounted or migrated by the production platform relocation.

Compose also starts **`inference-api`** and **`inference-worker`** from the repository-level [`inference/`](../inference/) package. That separate inference service owns contracts, registry-backed sync runs, and async job submission. `inference-api` is the HTTP boundary; `inference-worker` is reserved for slow CPU/GPU inference work. The platform API calls it through `INFERENCE_URL`. See [`inference/README.md`](../inference/README.md).

Useful after `-d`: `docker compose ps`, `docker compose logs -f`, `docker compose down`.

- `NEXT_PUBLIC_API_BASE_URL` (build/runtime env) — URL the browser uses (`http://localhost:8000`)

Rebuild the frontend image if you change `NEXT_PUBLIC_API_BASE_URL`.

### Bumping the Docker version

**Release source of truth:** [`VERSION`](VERSION) at the repo root of `nomicous/`.

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

**Important:** set `NOMICOUS_VERSION` in a shell profile or root `.env` next to
`docker-compose.yml` so the image tag always matches `VERSION`. The current
Compose fallback is `0.3.3`; do not rely on it for a release unless it has been
updated to match `VERSION`.

## Environment variables

| Variable | File | Default | Purpose |
|----------|------|---------|---------|
| `DATABASE_URL` | `backend/core/.env` | local Compose DB URL | Async platform database URL |
| `SYNC_DATABASE_URL` | `backend/core/.env` | local Compose DB URL | Alembic database URL |
| `JWT_SECRET` | `backend/core/.env` | development secret | Auth token signing key |
| `CORS_ORIGINS` | `backend/core/.env` | `http://localhost:3000,http://localhost:5173` | Allowed browser origins |
| `MEDIA_ROOT` | `backend/core/.env` | `nomicous/backend/media` | Uploaded document part media |
| `NEXT_PUBLIC_API_BASE_URL` | `frontend/.env.local` | `http://localhost:8000` | Frontend → API URL |

## Inference Catalog

`inference/registry.yaml` is the runtime model catalog. The development seed
creates `InferenceModel` rows with `registry://<model-id>?tag=stable` artifact
references and project-level bindings. Its defaults are
`kraken-segment` for segmentation and `syriac-calamari-v1` for
transcription. `greek-calamari-v1` stays commented out in `registry.yaml` until
Hub revision + artifact SHA are pinned.

```bash
uv run --group platform python scripts/platform/seed_dev_inference.py
```

Model bytes are resolved by the inference service from the registry's
`package://`, `hf://`, or optional `file://` source. See
[`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

## Tests (TDD)

```bash
# From repository root
uv run --group platform --group inference pytest tests/nomicous/unit
uv run --group platform --group inference pytest tests/nomicous/integration -m "not ml"
```

See [`docs/guides/testing.md`](../docs/guides/testing.md) for the ML lane and full-suite commands.

## OpenAPI

```bash
python scripts/platform/export_openapi.py

cd nomicous/frontend
npm run codegen:api
```

Generated schema types: `frontend/src/api/schema.d.ts`. App-facing aliases live
in `frontend/src/api/client.ts`.

