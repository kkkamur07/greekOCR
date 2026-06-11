# annote

AI-assisted transcription platform for manuscript pages. A standalone local tool for segmenting page images (manually or with Kraken auto-segmentation), pairing each segment with its transcription line, and exporting training-ready processed line images and text files.

See `CONTEXT.md` for domain glossary and `issues/prd.md` for requirements.

## Data layout

All persistence is under `data/` (path configurable via `ANNOTE_DATA_ROOT` in `backend/.env`):

| Path | Contents |
|------|----------|
| `data/manuscripts/pages/` | Source page images (`.jpg`, `.png`, …) |
| `data/transcriptions/pages/` | Page transcriptions (`<stem>.txt`, line-broken) |
| `data/annotations/pages/` | Per-page annotation JSON (segments + pairings; tool-internal) |
| `data/manuscripts/export/` | **Exported outputs** — paired `<stem>_<segment_number>.jpg` and `.txt` side by side |

Missing subdirectories are **created automatically** when the API starts. If creation fails (permissions, bad path), startup aborts with a clear error pointing at `ANNOTE_DATA_ROOT`.

### Sample fixture

`data/manuscripts/pages/sample_folio.jpg` and `data/transcriptions/pages/sample_folio.txt` are included for local testing.

## Setup

### Backend virtual environment

```bash
cd annote/backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,kraken for auto-segmentation
cp .env.example .env               # edit ANNOTE_DATA_ROOT / ANNOTE_PORT if needed
```

### Frontend

```bash
cd annote/frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL must match ANNOTE_PORT
npm install
```

## Development (two terminals)

### Terminal 1 — API

```bash
cd annote/backend
source .venv/bin/activate
annote                             # reads host/port/data root from .env
```

Defaults from `.env.example`: `http://127.0.0.1:5050`

### Terminal 2 — Frontend

```bash
cd annote/frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Docker (one command)

Ensure ports **3000** and **5050** are free. From `annote/`:

```bash
docker compose up --build      # foreground; Ctrl+C stops the stack
docker compose up --build -d   # detached background
```

| Service | URL |
|---------|-----|
| Frontend | [http://localhost:3000](http://localhost:3000) |
| API | [http://localhost:5050](http://localhost:5050) |

`data/` is mounted into the backend container, so pages, annotations, and exports persist on the host.

Useful after `-d`: `docker compose ps`, `docker compose logs -f`, `docker compose down`.

- `NEXT_PUBLIC_API_BASE_URL` (build arg) — URL the **browser** uses (`http://localhost:5050`)
- `API_INTERNAL_URL` (runtime env on the frontend service) — URL **server-side** Next.js uses inside Compose (`http://backend:5050`)

Rebuild the frontend image if you change `NEXT_PUBLIC_API_BASE_URL`.

### Bumping the Docker version

**Single source of truth:** [`VERSION`](VERSION) at the repo root of `annote/` (also used by the Python package via `pyproject.toml`).

1. Edit `VERSION` (semver, one line — e.g. `0.2.1`).
2. Export it and rebuild so Compose tags images correctly:

```bash
export ANNOTE_VERSION=$(cat VERSION)
docker compose up --build -d
```

`docker-compose.yml` builds `annote-backend:${ANNOTE_VERSION}` and `annote-frontend:${ANNOTE_VERSION}` and passes `APP_VERSION` into both Dockerfiles (API `/health` and frontend build).

3. Verify:

```bash
curl -s http://localhost:5050/health | python -m json.tool
docker images 'annote-*'
```

**When to bump:** any release you want to distinguish in image tags or `/health` — not required for every code change during dev (rebuild with the same `ANNOTE_VERSION` is fine).

**Optional:** set `ANNOTE_VERSION` in a shell profile or `.env` next to `docker-compose.yml` so you do not export it every time. If `ANNOTE_VERSION` is unset, Compose defaults to `0.1.0` — keep it in sync with `VERSION` when tagging releases.

## Environment variables

| Variable | File | Default | Purpose |
|----------|------|---------|---------|
| `ANNOTE_DATA_ROOT` | `backend/.env` | `data` (→ `annote/data`) | Filesystem data root; relative paths resolve from `annote/` |
| `ANNOTE_HOST` | `backend/.env` | `127.0.0.1` | API bind host |
| `ANNOTE_PORT` | `backend/.env` | `5050` | API bind port |
| `ANNOTE_CORS_ORIGINS` | `backend/.env` | `http://localhost:3000` | Allowed browser origins |
| `ANNOTE_RELOAD` | `backend/.env` | `true` | Uvicorn auto-reload |
| `NEXT_PUBLIC_API_BASE_URL` | `frontend/.env.local` | `http://localhost:5050` | Frontend → API URL |

## Tests (TDD)

```bash
cd annote/backend
source .venv/bin/activate
pytest
```

## OpenAPI

```bash
cd annote/backend
PYTHONPATH=. python scripts/export_openapi.py

cd ../frontend
npm run generate:openapi
```

Generated schema types: `frontend/src/types/openapi.ts`. App-facing aliases: `frontend/src/types/api.ts`.

