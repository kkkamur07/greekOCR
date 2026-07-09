# Local development

Run the full Nomicous stack on your machine. For production hosting, see [deployment/production.md](../deployment/production.md).

---

## Quick start (Docker Compose)

From the repository root:

```bash
cp nomicous/backend/core/.env.example nomicous/backend/core/.env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Platform API | http://localhost:8000 |
| API health | http://localhost:8000/health |
| OpenAPI | http://localhost:8000/docs |
| Inference API (cloud) | http://localhost:8010 |
| Local Inference Helper | http://127.0.0.1:8001 (if installed) |
| Postgres | `localhost:5433` — user `postgres`, password `dev`, database `kalamos` |

Migrations run automatically when the API container starts.

---

## Supabase instead of local Postgres

Use a hosted Supabase project for shared testing:

1. Copy `nomicous/backend/core/.env.supabase.example` → `.env.supabase` and fill credentials.
2. Run `./scripts/platform/migrate_supabase.sh`.
3. Start with the Compose override:

```bash
docker compose -f docker-compose.yml -f docker-compose.supabase.yml up --build
```

Full guide: [deployment/supabase.md](../deployment/supabase.md).

---

## Local API only (Postgres in Docker)

```bash
docker compose up db -d
cd nomicous
export PYTHONPATH=.
cp backend/core/.env.example backend/core/.env
uv run --project .. --group platform alembic -c infrastructure/alembic.ini upgrade head
uv run --project .. --group platform uvicorn backend.core.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend only

```bash
cd nomicous/frontend
cp .env.local.example .env.local
npm install
npm run dev
```

App: http://localhost:5173 — see [`nomicous/frontend/README.md`](../../nomicous/frontend/README.md).

---

## Inference Helper (local OCR)

```bash
uv run --group inference python -m inference.helper
```

Probes `http://127.0.0.1:8001`. Packaging and DMG install: [`packaging/helper/README.md`](../../packaging/helper/README.md).

---

## More detail

| Doc | Topic |
|-----|--------|
| [Root README](../../README.md) | Repo overview, training, domain model |
| [`nomicous/README.md`](../../nomicous/README.md) | App operations, env vars, version bumps |
| [`nomicous/infrastructure/README.md`](../../nomicous/infrastructure/README.md) | Alembic, database wiring |
| [`inference/README.md`](../../inference/README.md) | Inference service, registry, Compose ports |
| [testing.md](testing.md) | Pytest commands |
