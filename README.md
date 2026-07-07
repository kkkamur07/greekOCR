# greekOCR (Kalamos)

**Kalamos** helps researchers transcribe and annotate Greek Byzantine manuscripts — accurate text, with models and an editor in one repo.

Two main pieces:

1. **Recognition models** — train and evaluate line-level Greek OCR (Calamari, legacy TrOCR/Kraken experiments).
2. **Nomicous production app** — FastAPI + Postgres backend and a React editor under [`nomicous/`](nomicous/) (eScriptorium-style hierarchy: projects → documents → parts → layout → transcriptions).

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
  model/calamari/     # Vendored Calamari OCR library (canonical; see docs/calamari-vendored-architecture.md)
  model/kraken/       # Kraken finetuning helpers
  train/calamari/     # Hydra training entry points (train.py, finetune.py, finetune.sh)
  preprocessing_data/ # Dataset pack builders
configs/              # Training presets (calamari_train.yaml, kraken_seg.yaml, …)
data/                 # Labelled crops and dataset packs (e.g. labelledData/)
experiments/          # Jupyter notebooks (calamari.ipyn_greek.ipynb, trOCR.ipynb, …)
outputs/              # Checkpoints and logs (created by training; see configs/)

inference/           # FastAPI inference service (segment + transcribe)
  weights/            # Shipped model weights (Calamari, Kraken)
  registry.yaml       # Model catalog consumed by inference-api and inference-worker
```

Training entry points:

- **Calamari train** — `src/train/calamari/train.py` (Hydra config `configs/calamari_train.yaml`)
- **Calamari finetune** — `src/train/calamari/finetune.sh` (config `configs/calamari_finetune.yaml`)
- **Kraken finetune** — `src/model/kraken/finetuning.py` (config `configs/kraken_seg.yaml`)

Checkpoints default to `outputs/calamari-greek-bible/` (override via Hydra `output.root` in the config files). For production inference, weights live under `inference/weights/` and are registered in `inference/registry.yaml` — see [`inference/README.md`](inference/README.md). Calamari **code** is vendored under `src/model/calamari/` (not the PyPI package at runtime): [`docs/calamari-vendored-architecture.md`](docs/calamari-vendored-architecture.md).

---

## Nomicous Production App (API + Editor)

The production app lives under [`nomicous/`](nomicous/). Its API uses **domain-driven design**: `nomicous/backend/core/` wires routers; bounded contexts are `users`, `project`, `document`, `annotation`, `ml`, and `jobs`. Postgres and Alembic live in [`nomicous/infrastructure/`](nomicous/infrastructure/).

### Prerequisites

- Python 3.11+
- Docker (Postgres; optional full stack)
- Node.js 20+ (frontend)

### Quick start (Docker Compose)

Compose project name is **`nomicous`** (database **`kalamos`**, repo branding **greekOCR / Kalamos**). Run from the repository root:

```bash
cp nomicous/backend/core/.env.example nomicous/backend/core/.env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend (editor) | http://localhost:5173 |
| API | http://localhost:8000 |
| Health | http://localhost:8000/health |
| OpenAPI | http://localhost:8000/docs |
| ML inference API | http://localhost:8001 |
| Postgres | `localhost:5433` — user `postgres`, password `dev`, database **`kalamos`** |

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

Full suite (platform + ML service + unit), prerequisites, and known failure notes: [docs/testing.md](docs/testing.md).

More detail: [nomicous/infrastructure/README.md](nomicous/infrastructure/README.md) (DB, migrations), [nomicous/backend/core/README.md](nomicous/backend/core/README.md) (settings, DTOs, routes), and [nomicous/README.md](nomicous/README.md) (app operations).

### Frontend

```bash
cd nomicous/frontend
cp .env.local.example .env.local   # if needed
npm install
npm run dev
```

App: http://localhost:5173 — see [nomicous/frontend/README.md](nomicous/frontend/README.md) for OpenAPI codegen, auth, jobs panel, and public published view.

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

Contributors: see [`issues/`](issues/) (`kanban.md`, `dag.md`) for what to work on next.

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
| ML inference API | http://localhost:8001 |

Training code (`src/`), root `data/`, and inference weights (`inference/weights/`) are intentionally separate from the production app. Nomicous platform media is stored under `nomicous/backend/media/`.

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
