# Infrastructure (repo root)

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

From the **repository root** (deps only — repo is **not** installed as a package; `PYTHONPATH=.` is set via pytest config and below):

```bash
docker compose up db -d
uv venv && source .venv/bin/activate
uv pip install -r requirements/requirements-platform.txt
export PYTHONPATH=.   # needed for alembic / uvicorn from the shell
cp backend/core/.env.example backend/core/.env
alembic -c infrastructure/alembic.ini upgrade head
pytest -v
docker compose up --build   # optional full stack
```

Env and settings: `backend/core/.env` + [backend/core/settings/](../backend/core/settings/).
