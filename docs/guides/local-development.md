# Local development

Run the full Nomikos stack on your machine. For the concise user-facing
setup and hosting guide, see [using and hosting Nomikos](using-and-hosting.md).
This page keeps service-by-service contributor instructions. For production
hosting, see [deployment/production.md](../deployment/production.md).

---

## Quick start (Docker Compose)

From the repository root:

```bash
cp infrastructure/.env.compose.example infrastructure/.env
cp nomikos/backend/core/.env.example nomikos/backend/core/.env
docker compose -f infrastructure/docker-compose.yml up --build
```

| Service      | URL                                                                             |
| ------------ | ------------------------------------------------------------------------------- |
| Frontend     | http://localhost:5173                                                           |
| Platform API | http://localhost:8000                                                           |
| API health   | http://localhost:8000/health                                                    |
| OpenAPI      | http://localhost:8000/docs                                                      |
| Postgres     | `127.0.0.1:5433` - credentials from the ignored `infrastructure/.env`, database `kalamos` |

Migrations run automatically when the API container starts.

Compose publishes no inference port. Models run in an **inference agent** you
start on the host (below); it calls the platform, so it needs no address of its
own.

---

## Supabase instead of local Postgres

Use a hosted Supabase project for shared testing:

1. Copy `nomikos/backend/core/.env.supabase.example` → `.env.supabase` and fill credentials.
2. Follow the [database role runbook](../deployment/database-roles.md), then run `./scripts/platform/migrate_supabase.sh`.
3. Start with the Compose override:

```bash
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml up --build
```

Full guide: [deployment/supabase.md](../deployment/supabase.md).

---

## Local API only (Postgres in Docker)

```bash
docker compose -f infrastructure/docker-compose.yml up db -d
cd nomikos
export PYTHONPATH=.
cp backend/core/.env.example backend/core/.env
uv run --project .. --group platform alembic -c infrastructure/alembic.ini upgrade head
uv run --project .. --group platform uvicorn backend.core.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

---

## Frontend only

```bash
cd nomikos/frontend
cp .env.local.example .env.local
npm install
npm run dev
```

App: http://localhost:5173 - see [`nomikos/frontend/README.md`](../../nomikos/frontend/README.md).

---

## Inference agent (local OCR)

Against a local platform, from a source checkout:

```bash
NOMIKOS_API_URL=http://localhost:8000 uv run --group inference python -m nomikos_inference.cli pair
NOMIKOS_API_URL=http://localhost:8000 uv run --group inference python -m nomikos_inference.cli run
```

`pair` writes the device credential to `~/.nomikos/device.json`; `run` is the
**claim** loop - one page at a time, fetched through a short-lived signed link,
reported through the platform's job callback. It listens on nothing, so there
is no port to probe and nothing to add to a CORS or CSP allowlist.

`--exit-when-empty` stops the loop once the queue is empty, which is what to use
in a script; without it `run` waits for more work. `--api-url` overrides
`NOMIKOS_API_URL` per invocation.

Researchers install the same program from PyPI rather than running it from a
checkout - `uv tool install nomikos-inference`, then
`nomikos pair` and `nomikos run`
([`nomikos_inference/README.md`](../../nomikos_inference/README.md#install)). There is no
installer to build.

---

## More detail

| Doc                                                                            | Topic                                                             |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| [Root README](../../README.md)                                                 | Repo overview, training, domain model                             |
| [`nomikos/README.md`](../../nomikos/README.md)                               | App operations, env vars, version bumps                           |
| [`nomikos/infrastructure/README.md`](../../nomikos/infrastructure/README.md) | Alembic, database wiring                                          |
| [`nomikos_inference/README.md`](../../nomikos_inference/README.md)                             | Published package, CLI, registry, weights                         |
| [learnings.md](learnings.md)                                                   | Supabase, serverless (Vercel), Calamari training, frequent errors |
| [testing.md](testing.md)                                                       | Pytest commands                                                   |
