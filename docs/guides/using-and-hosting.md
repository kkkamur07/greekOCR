# Use and host Nomicous

Nomicous lets researchers upload manuscript pages, segment written lines,
generate model transcription drafts, correct them, collaborate, publish
selected documents, and export paired line images and text.

This guide covers the supported local stack and the current hosted topology.

## Docker Compose quick start

The repository’s Compose file is a development stack, not a hardened
internet-facing deployment. It uses local Postgres, local filesystem media,
bind mounts, reload mode, and development seed data.

Prerequisites:

- Git
- Docker Desktop with Compose
- about 10 GB of free disk space

From the repository root:

```bash
git clone https://github.com/kkkamur07/greekOCR.git
cd greekOCR
cp .env.compose.example .env
```

Replace the placeholders in `.env`:

```text
POSTGRES_PASSWORD
JWT_SECRET
INFERENCE_WEBHOOK_SECRET
INFERENCE_SERVICE_SECRET
```

Start the stack:

```bash
docker compose up --build
```

Open <http://localhost:5173>. Development seed credentials are
`dev@example.com` / `dev-pass-123`.

| Service               | Address                      |
| --------------------- | ---------------------------- |
| Editor                | <http://localhost:5173>      |
| Platform API          | <http://localhost:8000>      |
| API documentation     | <http://localhost:8000/docs> |
| Compose inference API | <http://localhost:8010>      |
| Postgres              | `127.0.0.1:5433`             |

The first inference request downloads public weights into `src/hf/cache`.
Host port `8010` maps to the inference container’s port `8001`; host port
`8001` is reserved for the optional local helper.

```bash
docker compose ps
curl -s http://localhost:8000/health | python -m json.tool
docker compose logs -f
docker compose down
```

## Run services individually

For service development, install Python 3.11–3.12, `uv`, Node.js 20+, and
Docker for Postgres:

```bash
uv sync --group platform --group inference
cp nomicous/backend/core/.env.example nomicous/backend/core/.env
docker compose up db -d
```

Start the inference API:

```bash
PYTHONPATH=. uv run --group inference \
  uvicorn inference.api.main:app --host 0.0.0.0 --port 8001 --reload
```

Start the platform API:

```bash
cd nomicous
uv run --project .. --group platform \
  alembic -c infrastructure/alembic.ini upgrade head
uv run --project .. --group platform \
  uvicorn backend.core.app:create_app --factory --reload --port 8000
```

Start the frontend:

```bash
cd nomicous/frontend
npm install
cp .env.local.example .env.local
npm run dev
```

This path does not start the asynchronous workers. Start the platform and
inference workers too when testing remote jobs.

## Local inference helper

The helper runs models on the researcher’s machine and caches them at
`~/.nomicous/hf/cache`. It has no platform database, project authorization, or
job queue.

```bash
HELPER_REGISTRY_URL=http://localhost:8000/inference/v1/registry \
HF_CACHE_ROOT=~/.nomicous/hf/cache \
uv run --group inference python -m inference.helper
```

Verify it:

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/inference/v1/catalog
```

For hosted use, install a platform-specific package from the
[latest GitHub release](https://github.com/kkkamur07/greekOCR/releases/latest).
The released helper accepts browser requests from `https://app.nomicous.com`.
Local frontend origins may require development-specific CORS configuration.
Never expose the helper beyond loopback without secure mode, a strong helper
secret, and TLS.

## Supabase-backed development

1. Create a Supabase project.
2. Create a private Storage bucket named `document-media`.
3. Fill:
   - `nomicous/backend/core/.env.supabase.example`
   - `nomicous/backend/core/.env.inference.example`
4. Set service secrets in the repository-root `.env`.
5. Provision the least-privilege roles in
   [`../deployment/database-roles.md`](../deployment/database-roles.md).
6. Run migrations with the direct migrator connection:

```bash
./scripts/platform/migrate_supabase.sh
```

7. Start the overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.supabase.yml up --build
```

Runtime traffic uses configured pooler connections; migrations use the direct
connection. Keep migration URLs and Storage service keys server-side.

## Hosted production

The current production topology is manual:

- Vercel: landing page, Next.js editor, and request/response FastAPI API;
- Supabase: Postgres and private `document-media` Storage;
- persistent Docker host: optional cloud inference, platform worker, and
  inference worker.

There is no complete one-click hosting template. Operators must configure DNS,
secrets, roles, migrations, TLS, backups, monitoring, and worker supervision.

The documented default is local inference through the user-installed helper.
Vercel cannot run long-lived PyTorch, Kraken, or Calamari workers. If remote
inference is enabled, all of these persistent processes are required:

```text
platform-worker   -> submits platform jobs
inference-api     -> accepts and queues inference jobs
inference-worker  -> runs models and calls the platform API
```

Cloud inference is disabled in the documented Vercel defaults. Enable it only
after deploying and verifying the complete worker and callback path.

See the detailed operator runbooks:

- [`../deployment/production.md`](../deployment/production.md)
- [`../deployment/supabase.md`](../deployment/supabase.md)
- [`../../deploy/inference/README.md`](../../deploy/inference/README.md)
