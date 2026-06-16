# Infrastructure (annote app root)

Postgres engine, Alembic migrations, and ORM metadata aggregation. **Settings** live in `backend/core/settings/`; this folder is DB + migrations only.

```text
infrastructure/
  db.py              # async engine, session, Base
  models.py          # imports all context ORM for Alembic metadata
  alembic.ini
  alembic/           # env.py reads InfrastructureSettings from backend.core.settings
```

Bounded contexts keep context ORM in `backend/<context>/infrastructure/orm_models.py`.

## Migrations (by domain)

Revisions under `infrastructure/alembic/versions/` — one file per bounded context (split for maintainability):

| Revision | Domain | Tables |
|----------|--------|--------|
| `001_users` | users | `users` |
| `002_project` | project | `projects`, `project_shared_users` |
| `003_inference_models` | inference (catalog) | `inference_models` |
| `004_document_layout` | document (layout) | `documents`, `document_parts`, `blocks`, `lines` |
| `005_inference_jobs` | inference (jobs) | `model_bindings`, `jobs` |
| `006_document_transcriptions` | document (text) | `transcriptions`, `line_transcriptions` |

`006` depends on `005` because `transcriptions.created_by_job_id` references `jobs`.

**If you already migrated the old single-file revision** (`e8d4f5814200`), reset dev data:

```bash
docker compose down
docker volume rm greekocr_postgres_data
docker compose up db -d
alembic -c infrastructure/alembic.ini upgrade head
```

## Database: `kalamos`

| Setting | Value (local dev) |
|---------|-------------------|
| Host | `localhost` (or `db` in Compose) |
| Port | `5433` → `5432` |
| User | `postgres` |
| Password | **`dev`** (local dev only — change in production) |
| Database | **`kalamos`** |

Connection URL (host): `postgresql://postgres:dev@localhost:5433/kalamos`

## Verify locally

From the **annote app root** (deps only — repo is **not** installed as a package; `PYTHONPATH=.` is set via pytest config and below):

```bash
docker compose up db -d
python -m venv .venv && source .venv/bin/activate
pip install -e "backend[dev]"
export PYTHONPATH=.   # needed for alembic / uvicorn from the shell
cp backend/core/.env.example backend/core/.env
alembic -c infrastructure/alembic.ini upgrade head
pytest backend/tests/platform -v
docker compose up --build   # optional full stack
```

Env and settings: `backend/core/.env` + [backend/core/settings/](../backend/core/settings/).

## Inference model catalog

Issue 005 keeps model weights outside the production app tree. For local development,
store Kraken artifacts under the repository-level `model/` workspace, for example:

```text
../model/kraken/
  segment.mlmodel
  transcribe.mlmodel
```

`backend/core/.env` controls the names and base path registered by the dev seed:

```bash
DEFAULT_SEGMENT_MODEL=kraken-segment-default
DEFAULT_TRANSCRIBE_MODEL=kraken-transcribe-default
KRAKEN_MODEL_PATH=../model/kraken
```

After `alembic -c infrastructure/alembic.ini upgrade head`, seed the catalog and
project-level default bindings from the annote app root:

```bash
PYTHONPATH=. python scripts/seed_dev_inference.py
```

The script upserts two `InferenceModel` rows (`segment`, `transcribe`) and a dev
project with project-level bindings. It records artifact paths only; it does not
download, create, or modify model weights. A later smoke-test fixture should use
`backend/media/fixtures/sample_folio.png` or another checked-in image path once
segment job tests land.
