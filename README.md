# greekOCR (Kalamos)

**Kalamos** helps researchers transcribe and annotate Greek Byzantine manuscripts — accurate text, with models and an editor in one repo.

Two main pieces:

1. **Recognition models** — train and evaluate line-level Greek OCR (Calamari, legacy TrOCR/Kraken experiments).
2. **Annote production app** — FastAPI + Postgres backend and a React editor under [`annote/`](annote/) (eScriptorium-style hierarchy: projects → documents → parts → layout → transcriptions).

---

## Recognition quality (Calamari)

We started with TrOCR on English-centric checkpoints — roughly **75% character error** on our line crops, fine for experiments, not for publication. Finetuning **Calamari** on labelled New Testament lines was the breakthrough.

On 410 held-out test images, the `calamari-greek-bible` checkpoint usually scores **under 2% CER** per line. Example from [`model/experiments/kalariOCR.ipynb`](model/experiments/kalariOCR.ipynb):

| | Text |
|---|------|
| **Ground truth** | Ἰωσαφὰτ δὲ ἐγέννησε τὸν Ἰωράμ. Ἰωρὰμ δὲ ἐγέννησε τὸν Ὀζίαν. |
| **Prediction** | Ἰωσαφὰτ δὲ ἐγέννησε τὸν Ἰωράμ. Ἰωρὰμ δὲ ἐγένησε τὸν Ὀζίαν. |
| **Metrics** | **CER 1.69%** · WER 10.00% |

One character differs in the second verb (*ἐγέννησε* vs *ἐγένησε*) — easy to correct in the editor.

**Pipeline:** Kraken (`blla.mlmodel`) for binarization and line segmentation → line crops → Calamari recognition. Segmentation was already strong; recognition was the hard part. Calamari is the default recognizer now.

To rerun the eval: open the notebook, set `CHECKPOINT` to `model/outputs/calamari-greek-bible/best.ckpt`, and execute the metric cells.

---

## `model/` layout

All modelling code, data, experiments, and training artifacts live under **`model/`**:

```text
model/
  ocr/              # Training & inference code (Calamari, TrOCR, Kraken pipeline helpers)
  configs/          # Training presets and pack paths
  data/             # Labelled crops and dataset packs (e.g. labelledData/, calamari packs)
  experiments/      # Jupyter notebooks (kalariOCR.ipynb, trOCR.ipynb, …)
  outputs/          # Checkpoints, logs, Slurm output, exported line JSON
```

Training entry points:

- **Calamari** — `model/ocr/calamari_ocr/train.sh` (env vars `CALAMARI_PACK`, `CALAMARI_OUTPUT`, …)
- **Pack builder** — `model/ocr/calamari_ocr/prepare_calamari_pack.py`
- **CLI** — `model/ocr/main.py`

Checkpoints default to `model/outputs/calamari-greek-bible/` (override with `CALAMARI_OUTPUT`).

---

## Annote Production App (API + Editor)

The production app lives under [`annote/`](annote/). Its API uses **domain-driven design**: `annote/backend/core/` wires routers; bounded contexts are `users`, `project`, `document`, `annotation`, `ml`, and `jobs`. Postgres and Alembic live in [`annote/infrastructure/`](annote/infrastructure/).

### Prerequisites

- Python 3.11+
- Docker (Postgres; optional full stack)
- Node.js 20+ (frontend)

### Quick start (Docker Compose)

```bash
cp annote/backend/core/.env.example annote/backend/core/.env
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Health | http://localhost:8000/health |
| OpenAPI | http://localhost:8000/docs |
| Postgres | `localhost:5433` — user `postgres`, password `dev`, database **`kalamos`** |

Migrations run on API container start (`alembic upgrade head`).

### Local API (DB in Docker only)

```bash
docker compose up db -d
cd annote
export PYTHONPATH=.
cp backend/core/.env.example backend/core/.env
uv run --project ../ --group platform alembic -c infrastructure/alembic.ini upgrade head
uv run --project ../ --group platform uvicorn backend.core.main:app --reload --host 0.0.0.0 --port 8000
uv run --project ../ --group platform pytest backend/tests/platform
```

More detail: [annote/infrastructure/README.md](annote/infrastructure/README.md) (DB, migrations), [annote/backend/core/README.md](annote/backend/core/README.md) (settings, DTOs, routes), and [annote/README.md](annote/README.md) (app operations).

### Frontend

```bash
cd annote/frontend
cp .env.local.example .env.local   # if needed
npm install
npm run dev
```

App: http://localhost:5173 — see [annote/frontend/README.md](annote/frontend/README.md) for OpenAPI codegen, auth, jobs panel, and public published view.

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

[`annote/`](annote/) also contains the manuscript annotation workflow: segmenting Page images, pairing each Segment with Ground truth transcription, generating Transcription PDFs, and exporting training-ready Processed line images plus Line transcription files. Processing is pluggable: today the main step is **polygon rectification** (mask → axis-aligned crop), but the pipeline can be extended without changing the editor.

### Quick start (Docker)

From the repository root:

```bash
docker compose up --build      # foreground (logs in terminal; Ctrl+C stops)
docker compose up --build -d   # detached (runs in background)
```

After a detached start: `docker compose ps`, `docker compose logs -f`, `docker compose down`.

| Service | URL |
|---------|-----|
| Editor | http://localhost:3000 |
| API | http://localhost:8000 |

The repository root `model/` workspace and existing root `data/` contents are intentionally separate from the production app. Annote platform media is stored under `annote/backend/media/`.

### Bumping the Docker image version

1. Edit [`annote/VERSION`](annote/VERSION) (e.g. `0.2.0` → `0.2.1`).
2. Rebuild and tag images with that version:

```bash
export ANNOTE_VERSION=$(cat annote/VERSION)
docker compose up --build -d
```

Compose tags images as `annote-api:$ANNOTE_VERSION` and `annote-frontend:$ANNOTE_VERSION`. Confirm with `curl http://localhost:8000/health` (`version` in the JSON). Full details: [annote/README.md](annote/README.md).


### Note :
Have been using `OTSU` to tighten the segments, but it doesn't seem to generalize -> which is not good but the `kraken` segmentation the default model, generalizes super well but needs minor finetuning`
