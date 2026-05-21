# Shared infrastructure

Postgres connection, settings, and Alembic metadata live here only — not under bounded contexts (`users/`, `project/`, etc.).

| File | Role |
|------|------|
| `config.py` | `DATABASE_URL`, `SYNC_DATABASE_URL`, JWT, CORS, `MEDIA_ROOT` |
| `db.py` | Async SQLAlchemy engine, `Base`, `get_db()` |
| `models.py` | Imports all context ORM modules so Alembic sees the full schema |

Migrations: `backend/alembic/` (configured from repo-root `alembic.ini`).

## Database: `kalamos`

Development Postgres database name is **`kalamos`** (Greek *κάλαμος* — reed pen used for writing on papyrus). Same name in Docker Compose, `.env`, and Alembic.

| Setting | Value (local dev) |
|---------|-------------------|
| Host | `localhost` (or `db` inside Compose) |
| Port | `5433` (host) → `5432` (container) |
| User / password | `postgres` / `dev` |
| Database | **`kalamos`** |

## Branch strategy (one issue = one branch)

Each parallel **AFK lane** gets its own branch off `main`, so reviews stay small and independent:

| Issue | Branch (example) | Lane |
|-------|------------------|------|
| 000 platform foundation | `feat/000-platform-foundation` | — |
| 001 user auth | `feat/001-user-auth-jwt` | A |
| 004 job runner | `feat/004-job-runner` | B |

Do **not** stack unrelated issues on one branch. Merge `000` before starting `001` and `004` in parallel.

**Current:** `feat/000-platform-foundation` is pushed — open a PR from that branch → `main`, review, merge, then pull `main` before creating the next feature branches.

## Verify locally

From the **repository root**:

```bash
# 1) Postgres only
docker compose up db -d

# 2) Python env + platform deps
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3) Env (copy once; uses database kalamos on port 5433)
cp backend/.env.example backend/.env

# 4) Migrations + tests
alembic upgrade head
pytest -v

# 5) Optional: full stack (API + DB; migrations run on API start)
docker compose up --build
```

**Checks**

- Health: http://localhost:8000/health → `{"status":"ok","database":"ok"}`
- OpenAPI: http://localhost:8000/docs

**Run API without rebuilding Docker** (DB still via Compose):

```bash
docker compose up db -d
source .venv/bin/activate
alembic upgrade head
uvicorn backend.core.main:app --reload --host 0.0.0.0 --port 8000
```
