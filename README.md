# greekOCR (Kalamos)

**Kalamos** helps researchers transcribe and annotate Greek Byzantine manuscripts — accurate text, with models and an editor in one repo.

Two main pieces:

1. **Recognition models** — train and evaluate line-level Greek OCR (Calamari, legacy TrOCR/Kraken experiments).
2. **Nomicous production app** — FastAPI + Postgres backend and a React editor under [`nomicous/`](nomicous/) (eScriptorium-style hierarchy: projects → documents → parts → layout → transcriptions).

**Documentation:** [`docs/README.md`](docs/README.md) — guides, deployment,
inference, ADRs, [repository hygiene](docs/repository-hygiene.md).

---

## Recognition quality (Calamari)

We started with TrOCR on English-centric checkpoints — roughly **75% character error** on our line crops, fine for experiments, not for publication. Finetuning **Calamari** on labelled New Testament lines was the breakthrough.

On 410 held-out test images, the `calamari-greek-bible` checkpoint usually scores **under 2% CER** per line. Example from [`experiments/calamari.ipyn_greek.ipynb`](experiments/calamari.ipyn_greek.ipynb):

| | Text |
|---|------|
| **Ground truth** | Ἰωσαφὰτ δὲ ἐγέννησε τὸν Ἰωράμ. Ἰωρὰμ δὲ ἐγέννησε τὸν Ὀζίαν. |
| **Prediction** | Ἰωσαφὰτ δὲ ἐγέννησε τὸν Ἰωράμ. Ἰωρὰμ δὲ ἐγένησε τὸν Ὀζίαν. |
| **Metrics** | **CER 1.69%** · WER 10.00% |

One character differs in the second verb (*ἐγέννησε* vs *ἐγένησε*) — easy to correct in the editor.

**Pipeline:** Kraken (`blla.mlmodel`) for binarization and line segmentation → line crops → Calamari recognition. Segmentation was already strong; recognition was the hard part. Calamari is the default recognizer now.

To rerun the eval: open the notebook, set `CHECKPOINT` to `outputs/calamari-greek-bible/best.ckpt`, and execute the metric cells.

---

## Training (`src/`) and inference (`inference/`)

Recognition work is split between **training** at the repo root and **inference** in a separate service:

```text
src/
  model/calamari/     # Vendored Calamari OCR library (canonical; see docs/guides/learnings.md#calamari-training)
  model/kraken/       # Kraken finetuning helpers
  train/calamari/     # Hydra training entry points (train.py, finetune.py, finetune.sh)
  preprocessing_data/ # Dataset pack builders
configs/              # Training presets (calamari_train.yaml, kraken_seg.yaml, …)
data/                 # Labelled crops and dataset packs (e.g. labelledData/)
experiments/          # Jupyter notebooks (calamari.ipyn_greek.ipynb, trOCR.ipynb, …)
outputs/              # Checkpoints and logs (created by training; see configs/)

inference/           # FastAPI inference service (segment + transcribe)
  registry.yaml       # Model catalog consumed by inference-api and inference-worker
```

Training entry points:

- **Calamari train** — `src/train/calamari/train.py` (Hydra config `configs/calamari_train.yaml`)
- **Calamari finetune** — `src/train/calamari/finetune.sh` (config `configs/calamari_finetune.yaml`)
- **Kraken finetune** — `src/model/kraken/finetuning.py` (config `configs/kraken_seg.yaml`)

Checkpoints default to `outputs/calamari-greek-bible/` (override via Hydra `output.root` in the config files). For production inference, weights are published to Hugging Face Hub and registered in `inference/registry.yaml` — see [`inference/README.md`](inference/README.md). Calamari **training** code is vendored under `src/model/calamari/`: [`docs/guides/learnings.md`](docs/guides/learnings.md#calamari-training).

---

## Nomicous Production App (API + Editor)

The production app lives under [`nomicous/`](nomicous/). Its API uses **domain-driven design**: `nomicous/backend/core/` wires routers; bounded contexts are `users`, `project`, `document`, `annotation`, `ml`, and `jobs`. Postgres and Alembic live in [`nomicous/infrastructure/`](nomicous/infrastructure/).

### Database (Supabase + Alembic)

For **shared testing and hosted deployments**, we use **Supabase** as managed Postgres (and Storage for page images). **Alembic** remains the single schema source of truth — the same migration history runs against local Docker Postgres and remote Supabase. We do **not** use Supabase Auth, PostgREST, or Supabase CLI migrations; the FastAPI app connects with SQLAlchemy and keeps its own JWT auth.

| Environment | Postgres | Page images |
|-------------|----------|-------------|
| Local dev (default) | Docker Compose `db` on `localhost:5433` | `MEDIA_ROOT` on disk (`STORAGE_BACKEND=local`) |
| Shared test / staging / production | Supabase hosted Postgres | Supabase Storage bucket (`STORAGE_BACKEND=supabase`) |

```bash
cp nomicous/backend/core/.env.supabase.example nomicous/backend/core/.env.supabase
# Follow docs/deployment/database-roles.md before the first migration.
./scripts/platform/migrate_supabase.sh
docker compose -f docker-compose.yml -f docker-compose.supabase.yml up --build
```

Operational guide: [`docs/deployment/supabase.md`](docs/deployment/supabase.md). Pitfalls (pooler + asyncpg, URL encoding, RLS): [`docs/guides/learnings.md`](docs/guides/learnings.md#supabase-hosted-postgres--storage).

### Prerequisites

- Python 3.11+
- Docker (Postgres; optional full stack)
- Node.js 20+ (frontend)

### Quick start (Docker Compose)

Compose project name is **`nomicous`** (database **`kalamos`**, repo branding **greekOCR / Kalamos**). Run from the repository root:

```bash
cp .env.compose.example .env
cp nomicous/backend/core/.env.example nomicous/backend/core/.env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend (editor) | http://localhost:5173 |
| API | http://localhost:8000 |
| Health | http://localhost:8000/health |
| OpenAPI | http://localhost:8000/docs |
| ML inference API | http://localhost:8010 (host port; container listens on 8001) |
| Postgres | `127.0.0.1:5433` — credentials are in the ignored root `.env`, database **`kalamos`** |

Also started: **`inference-worker`** (background inference jobs; no host port). The platform API calls `inference-api` via `INFERENCE_URL`. See [`inference/README.md`](inference/README.md).

Migrations run on API container start (`alembic upgrade head`).

### Local API (DB in Docker only)

```bash
docker compose up db -d
cd nomicous
export PYTHONPATH=.
cp backend/core/.env.example backend/core/.env
uv run --project ../ --group platform alembic -c infrastructure/alembic.ini upgrade head
uv run --project ../ --group platform uvicorn backend.core.main:app --reload --host 0.0.0.0 --port 8000
uv run --project ../ --group platform pytest tests/nomicous -q
```

Full suite (platform + ML service + unit), prerequisites, and known failure notes: [docs/guides/testing.md](docs/guides/testing.md).

More detail: [nomicous/infrastructure/README.md](nomicous/infrastructure/README.md) (DB, migrations), [nomicous/backend/core/README.md](nomicous/backend/core/README.md) (settings, DTOs, routes), and [nomicous/README.md](nomicous/README.md) (app operations).

### Frontend

```bash
cd nomicous/frontend
cp .env.local.example .env.local   # if needed
npm install
npm run dev
```

App: http://localhost:5173 — see [nomicous/frontend/README.md](nomicous/frontend/README.md) for OpenAPI codegen, auth, jobs panel, and public published view.

### Job status (SSE with polling fallback)

Segment and transcribe jobs are **product jobs** tracked in Postgres (`jobs` table). The browser prefers **Server-Sent Events (SSE)** and falls back to polling when a stream is unavailable, closes, or becomes idle.

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| Status change | Postgres `NOTIFY` on `platform_jobs` | API processes detect `jobs.status` updates after commit |
| Browser delivery | `GET /jobs/{job_id}/events` (`text/event-stream`) | Push `JobResponse` JSON as status moves `running` → `waiting` → `done` / `failed` |
| Fallback | `GET /jobs/{job_id}` on an interval | Used when SSE is unavailable (e.g. serverless API with `JOB_SSE_NOTIFICATIONS_ENABLED=false`) |

We implemented SSE to **cut redundant poll traffic**: a 45 s segmentation job at 250 ms polling was ~180 auth + DB round-trips that returned unchanged status. SSE wakes the UI as soon as the row commits and keeps API logs readable during long ML work.

Implementation: `nomicous/backend/jobs/infrastructure/notifications.py`, `nomicous/backend/jobs/api/jobs.py`, and frontend `subscribeToJob` in `nomicous/frontend/src/utils/jobSubscription.ts`. More detail: [nomicous/backend/README.md](nomicous/backend/README.md#job-status-notifications).

### Local inference helper

Researchers use a **hosted** web app, but segment and transcribe can run on **their laptop CPU** via the **Inference Helper** — a small background app (macOS, Windows, Linux) installed once. The browser cannot start Python on the user's machine; the hosted API also **cannot** call `localhost` on a researcher's laptop. So local inference is **browser-orchestrated**: the SPA talks to the helper, then persists results through the normal authenticated API.

```
┌─────────────┐     HTTPS      ┌──────────────┐     ┌──────────┐
│   Browser   │ ─────────────► │  Hosted API  │ ──► │ Postgres │
│  (hosted    │   persist      │  + auth JWT  │     │          │
│   SPA)      │ ◄───────────── │              │     └──────────┘
└──────┬──────┘                └──────────────┘
       │
       │ HTTP (127.0.0.1 only)
       ▼
┌─────────────────────┐
│  Inference Helper   │  localhost:8001
│  Calamari + Kraken  │  weights → ~/.nomicous/hf/cache/
└─────────────────────┘
```

| Path | Flow |
|------|------|
| **Cloud** (unchanged) | Browser enqueues a product job → `inference-worker` → webhook callback → merge into Postgres |
| **Local** | Browser → helper `POST /inference/v1/run` → browser → `POST …/local-inference/{segment,transcribe}` (JWT) → merge into Postgres |

The helper exposes only `GET /health`, `GET /inference/v1/catalog`, and `POST /inference/v1/run`. No Postgres, no job queue, no platform code. It binds to `127.0.0.1` and syncs `registry.yaml` from `GET /inference/v1/registry` on startup so new models do not require reinstalling the helper (weights still download on first use).

| Concept | Meaning |
|---------|---------|
| `host_eligibility: local` | Model may run on the helper when it is healthy |
| `host_eligibility: remote` | Cloud inference only |
| Cloud fallback | User toggles "Use cloud inference", or has no helper → existing remote job path |

**Default:** local when the helper is healthy and the model is `host_eligibility: local`.

Run locally for development:

```bash
HELPER_REGISTRY_URL=http://localhost:8000/inference/v1/registry \
HF_CACHE_ROOT=~/.nomicous/hf/cache uv run --group inference python -m inference.helper
```

Packaging (PyInstaller, auto-start): [`packaging/helper/README.md`](packaging/helper/README.md). Service detail: [`inference/README.md`](inference/README.md#inference-helper-local-cpu-on-researcher-machines).

### Production hosting

Production is live on **Vercel** (landing, SPA, platform API) + **Supabase** (Postgres + page image Storage) + **persistent Docker hosts** (inference API/worker, platform job worker). A future **all-Docker** deployment on our own server is planned; the same codebase already supports both modes via env flags.

| Surface | Host today | Notes |
|---------|------------|-------|
| `nomicous.com` | Vercel static (`landing/`) | Marketing site |
| `app.nomicous.com` | Vercel (`nomicous/frontend/`) | React SPA |
| `api.nomicous.com` | Vercel Python (`deploy/platform/`) | Serverless FastAPI → Supabase |
| `inference.nomicous.com` | Docker (Railway/Fly or similar) | PyTorch, long jobs, Postgres `LISTEN` |
| Platform + inference workers | Same persistent Docker host | Not serverless |

Runbook: [`docs/deployment/production.md`](docs/deployment/production.md).

#### Serverless API constraints (Vercel)

The platform API on Vercel is **request/response only** — no long-lived background tasks. Verified in production.

| Variable | Vercel value | Why |
|----------|--------------|-----|
| `JOB_WORKER_ENABLED` | `false` | Job dispatch runs on `platform-worker` (persistent Docker process) |
| `JOB_SSE_NOTIFICATIONS_ENABLED` | `false` | Postgres `NOTIFY` listener needs a persistent process |
| `BEHIND_PROXY` | `true` | Vercel terminates TLS; trust `X-Forwarded-*` headers |
| `STORAGE_BACKEND` | `supabase` | No local filesystem on serverless |

When SSE is off, the frontend **falls back to HTTP polling** for job status (`GET /jobs/{id}`) — jobs still complete; UI updates are slightly slower than Docker Compose where SSE is enabled.

Inference (PyTorch, 30+ minute jobs, model weights) **cannot** run on Vercel. ML stays on Docker; see [`deploy/inference/README.md`](deploy/inference/README.md).

Pitfalls and troubleshooting: [`docs/guides/learnings.md`](docs/guides/learnings.md#serverless-api-vercel).

---

## Domain model (eScriptorium-aligned)

| Entity | Role |
|--------|------|
| **Project** | Workspace; sharing via users |
| **Document** | Manuscript; workflow (draft / published / archived) |
| **DocumentPart** | One page image (ordered) |
| **Block** | Region on a page |
| **Line** | Text line with baseline geometry |
| **Transcription** | Named layer on a document |
| **LineTranscription** | Text (+ confidence) per line |
| **InferenceModel / Job** | Model catalog, bindings, async jobs |

The editor reuses interaction patterns from [eScriptorium](https://github.com/PSL-Paris-Saclay/escriptorium) (canvas, transcription panel, workflow) in React rather than a full Vue port.

---

## What we are building toward

- Curated, shareable corpora of legal and biblical Greek manuscripts
- Models that researchers can actually trust on their material
- Expert-in-the-loop annotation — compare runs, correct lines, publish read-only views

Contributors: see [`issues/README.md`](issues/README.md) for backlog and [`issues/kanban.md`](issues/kanban.md) for board state.

---

## Annotation + Custom Export

[`nomicous/`](nomicous/) also contains the manuscript annotation workflow: segmenting Page images, pairing each Segment with Ground truth transcription, generating Transcription PDFs, and exporting training-ready Processed line images plus Line transcription files. Processing is pluggable: today the main step is **polygon rectification** (mask → axis-aligned crop), but the pipeline can be extended without changing the editor.

### Quick start (Docker)

From the repository root:

```bash
docker compose up --build      # foreground (logs in terminal; Ctrl+C stops)
docker compose up --build -d   # detached (runs in background)
```

After a detached start: `docker compose ps`, `docker compose logs -f`, `docker compose down`.

| Service | URL |
|---------|-----|
| Editor | http://localhost:5173 |
| API | http://localhost:8000 |
| ML inference API | http://localhost:8010 (host port; container listens on 8001) |

Training code (`src/`), root `data/`, and Hub weight cache (`src/hf/cache/`) are intentionally separate from the production app. Nomicous platform media is stored under `nomicous/backend/media/`.

### Bumping the Docker image version

1. Edit [`nomicous/VERSION`](nomicous/VERSION) (e.g. `0.2.0` → `0.2.1`).
2. Rebuild and tag images with that version:

```bash
export NOMICOUS_VERSION=$(cat nomicous/VERSION)
docker compose up --build -d
```

Compose tags images as `nomicous-api:$NOMICOUS_VERSION` and `nomicous-frontend:$NOMICOUS_VERSION`. Confirm with `curl http://localhost:8000/health` (`version` in the JSON). Full details: [nomicous/README.md](nomicous/README.md).


### Note :
Have been using `OTSU` to tighten the segments, but it doesn't seem to generalize -> which is not good but the `kraken` segmentation the default model, generalizes super well but needs minor finetuning`
