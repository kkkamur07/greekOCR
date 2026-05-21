# Infrastructure (repo root)

Postgres, Alembic, and shared DB settings live here — **not** under `backend/`. Bounded contexts keep only context-specific ORM/repos in `backend/<context>/infrastructure/`.

```text
infrastructure/
  config.py          # DATABASE_URL, SYNC_DATABASE_URL, JWT, CORS
  db.py              # async engine, session, Base
  models.py          # imports all backend context ORM for Alembic metadata
  alembic.ini        # Alembic config (run with -c infrastructure/alembic.ini)
  alembic/           # env.py, versions/
  .env.example
```

## Database: `kalamos`

Development Postgres database name is **`kalamos`** (Greek *κάλαμος* — reed pen). Same name in Docker Compose, `.env`, and Alembic.

| Setting | Value (local dev) |
|---------|-------------------|
| Host | `localhost` (or `db` inside Compose) |
| Port | `5433` (host) → `5432` (container) |
| User / password | `postgres` / `dev` |
| Database | **`kalamos`** |

## Branch strategy (one issue = one branch)

Each parallel **AFK lane** gets its own branch off `main`:

| Issue | Branch | Lane |
|-------|--------|------|
| 000 platform foundation | `feat/000-platform-foundation` | — |
| 001 user auth | `feat/001-user-auth-jwt` | A |
| 004 job runner | `feat/004-job-runner` | B |

## Verify locally

From the **repository root**:

```bash
# 1) Postgres only
docker compose up db -d

# 2) Python env + platform deps
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3) Env (copy once; database kalamos on port 5433)
cp infrastructure/.env.example infrastructure/.env

# 4) Migrations + tests
alembic -c infrastructure/alembic.ini upgrade head
pytest -v

# 5) Optional: full stack (API + DB; migrations run on API start)
docker compose up --build
```

**Checks**

- Health: http://localhost:8000/health → `{"status":"ok","database":"ok"}`
- OpenAPI: http://localhost:8000/docs

**Run API without rebuilding Docker**:

```bash
docker compose up db -d
source .venv/bin/activate
alembic -c infrastructure/alembic.ini upgrade head
uvicorn backend.core.main:app --reload --host 0.0.0.0 --port 8000
```
