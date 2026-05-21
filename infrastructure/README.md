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
uv pip install -r requirements-platform.txt
export PYTHONPATH=.   # needed for alembic / uvicorn from the shell
cp backend/core/.env.example backend/core/.env
alembic -c infrastructure/alembic.ini upgrade head
pytest -v
docker compose up --build   # optional full stack
```

Env and settings: `backend/core/.env` + [backend/core/settings/](../backend/core/settings/).
