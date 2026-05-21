---
id: "000"
title: "platform-foundation"
type: AFK
status: review
blocked_by: []
parent_prd: "issues/prd.md"
branch: feat/000-platform-foundation
---

## Parent PRD

`issues/prd.md` — Implementation Decisions (Architectural shape, Domain persistence, Stack)

## What to build

End-to-end platform skeleton: Docker Compose (Postgres + API), bounded-context DDD layout, FastAPI app in `backend/core/`, `.env.example`. Demoable via `docker compose up`, migrated DB, `GET /health` and OpenAPI at `/docs`.

**Shared platform (`backend/infrastructure/`)** — Postgres + Alembic live here only (not under `project/` or any context):

```text
backend/infrastructure/
  config.py          # DATABASE_URL, SYNC_DATABASE_URL
  db.py              # async engine, session, Base
  models.py          # imports all context ORM modules for Alembic metadata
backend/alembic/     # env.py, versions/ — migrations for full schema
```

Context `infrastructure/` folders hold **context ORM/repos only** (e.g. `project/infrastructure/orm_models.py`), not the global DB connection.

## Acceptance criteria

- [x] `docker compose` starts Postgres and API; README or issue notes document ports and env vars
- [x] `backend/infrastructure/db.py` + `config.py` own Postgres connection; no duplicate engine in context packages
- [x] `backend/alembic/` configured with `env.py` using `SYNC_DATABASE_URL` and `backend.infrastructure.models` metadata
- [x] Alembic initial migration creates all v1 tables from PRD schema (including Job, InferenceModel, ModelBinding, manual_geometry flags)
- [x] FastAPI app in `backend/core/` boots with settings from env; health endpoint returns OK when DB is reachable
- [x] `alembic upgrade head` runs cleanly on empty database
- [x] DDD folders exist: `backend/core/` plus users, project, document, inference (each: domain / application / infrastructure / api)

## Blocked by

None — can start immediately.

## User stories addressed

- 50 (bounded contexts)
- 51 (Alembic migrations)
- 52 (Docker Compose local dev)

## Verification

```bash
docker compose up db -d
source .venv/bin/activate && uv pip install -e ".[dev]"
alembic upgrade head
pytest -v
```
