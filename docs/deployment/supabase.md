# Supabase deployment profile

Use Supabase as **managed Postgres + object storage** for shared test/staging deploys.
Local Docker development stays unchanged (`STORAGE_BACKEND=local`, Compose Postgres).

This document covers setup, configuration choices, known pitfalls, and trade-offs.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Browser (frontend)                                              │
│    → Platform API (FastAPI, local or hosted)                     │
└─────────────────────────────────────────────────────────────────┘
         │                                        ▲
         │ JWT auth (app-owned)                   │ claim a page, report it
         ▼                                        │ (outbound, device token)
┌─────────────────────┐              ┌──────────────────────────┐
│ Supabase Postgres   │              │ nomikos inference agent │
│ (postgres DB)       │              │ (researcher's machine)   │
└─────────────────────┘              └──────────────────────────┘
         │                                        │
         ▼                                        │ signed page image link
┌─────────────────────┐                           │
│ Supabase Storage    │ ◄─────────────────────────┘
│ bucket document-media│  ← page images only (WebP)
└─────────────────────┘
```

The browser never talks to the agent. The agent connects out to the platform,
claims one page, and fetches that page's image through a short-lived **signed
page image link** (ADR 0002).

### What goes where

| Layer    | Supabase product                | Contents                                                          |
| -------- | ------------------------------- | ----------------------------------------------------------------- |
| Database | Postgres (`postgres` DB)        | Users, projects, documents, layout, transcriptions, jobs, history |
| Storage  | Private bucket `document-media` | Document part **page images only** (WebP)                         |

**Not stored in Supabase Storage:** exports (PDF/XML), model weights, annotation JSON (that lives in Postgres).

### What we use from Supabase

| Product                   | Used?  | How                                           |
| ------------------------- | ------ | --------------------------------------------- |
| Postgres                  | Yes    | Direct SQL via SQLAlchemy + Alembic           |
| Storage                   | Yes    | Server-side via **secret (service role) key** |
| Auth                      | **No** | App JWT (`JWT_SECRET`)                        |
| Data API / PostgREST      | **No** | Backend talks SQL directly                    |
| Realtime / Edge Functions | **No** | -                                             |

**Alembic** remains the schema source of truth. The Supabase CLI is **not** used for migrations.

---

## Dashboard settings at project creation

When Supabase asks about Data API and RLS defaults:

| Setting                             | Recommendation             | Pros                                                                   | Cons                                                                 |
| ----------------------------------- | -------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Enable Data API**                 | **Off** (or On but unused) | Smaller attack surface; matches our stack (no `supabase-js` DB access) | Cannot use PostgREST / client SDK against tables without re-enabling |
| **Automatically expose new tables** | **Disable**                | Alembic tables stay private; no accidental `anon` access to new tables | Must grant manually if you later want Data API                       |
| **Enable automatic RLS**            | **Disable**                | Matches app-layer auth and the consolidated baseline                   | No DB-level row isolation if API is compromised                      |

We authorize in **FastAPI**, not Postgres RLS. The backend connects with the database password / service credentials, not the publishable key.

---

## API keys: publishable vs secret

| Dashboard label | Old name       | Use in Nomikos                                     |
| --------------- | -------------- | --------------------------------------------------- |
| **Publishable** | `anon`         | **Not used** - frontend talks to our API only       |
| **Secret**      | `service_role` | **`SUPABASE_SERVICE_ROLE_KEY`** in backend env only |

|          | Secret key                                                     | Publishable key                                                       |
| -------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Pros** | Full Storage access; server can upload/read/delete page images | Safe to embed in browsers; limited scope                              |
| **Cons** | Must never leak to frontend or git                             | Cannot manage private Storage server-side; wrong tool for our backend |

```bash
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<secret key from dashboard>
```

---

## Database connection strings

Supabase provides connection URIs in **Project Settings → Database**. Use
separate provider-managed service principals where the plan supports them; see
[database-roles.md](database-roles.md). Their credentials stay in provider
secrets, never in an example file or command history.

| Variable                | Connection                         | Port        | Driver                     | Purpose                                              |
| ----------------------- | ---------------------------------- | ----------- | -------------------------- | ---------------------------------------------------- |
| `MIGRATOR_DATABASE_URL` | **Direct** `db.<ref>.supabase.co`  | 5432        | `postgresql://` (psycopg2) | Alembic operator/migrator only                       |
| `DATABASE_URL`          | **Transaction pooler** `…pooler…`  | 6543        | `postgresql+asyncpg://`    | `nomikos_api` or platform-worker runtime            |
| `SYNC_DATABASE_URL`     | **Transaction pooler** (or direct) | 6543 / 5432 | `postgresql://`            | Matching runtime principal for sync listener/scripts |

Database name is **`postgres`** (Supabase default) - not `kalamos`.

### Assign service URLs

Copy the direct migrator URI only into the migration runner's secret store.
Copy an API principal's pooler URI into `DATABASE_URL` and
`SYNC_DATABASE_URL`; only `DATABASE_URL` adds `+asyncpg`. An inference agent
needs no database URI. Add `?sslmode=require` to libpq URLs when the provider
URI omits it.

### Direct vs transaction pooler

|          | Direct (`:5432`)                                            | Transaction pooler (`:6543`)                                         |
| -------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| **Pros** | Full Postgres features; Alembic DDL; prepared statements OK | Many short-lived connections; good for serverless / high concurrency |
| **Cons** | Connection limits on free tier; one session per connection  | No prepared statements with asyncpg (see below); not for migrations  |

**Rule:** migrations → direct; app runtime → pooler.

### Session pooler (alternative)

Supabase also offers **session mode** on pooler port `5432`. It supports prepared statements but pools less aggressively. Use if you cannot disable statement caching and need pooler semantics.

### Password special characters

Characters like `@`, `#`, `%` in the password **break URL parsing**. URL-encode them in the URI:

| Character | Encoded |
| --------- | ------- |
| `@`       | `%40`   |
| `#`       | `%23`   |
| `%`       | `%25`   |

Example: password `@Krrish@2021` → `%40Krrish%402021` in the URL.

Alembic also needs `%` doubled (`%%`) when passed through ConfigParser - handled in `infrastructure/alembic/env.py`.

---

## Environment files

| File                                           | Purpose                                        |
| ---------------------------------------------- | ---------------------------------------------- |
| `nomikos/backend/core/.env`                   | Default local dev (Docker Postgres)            |
| `nomikos/backend/core/.env.supabase`          | Supabase profile (gitignored)                  |
| `nomikos/backend/core/.env.supabase.example`  | Template (committed)                           |

Settings load **`.env` first**; if missing, fall back to **`.env.supabase`** (`backend/core/settings/_env.py`).

### Options

| Approach                         | Pros                                       | Cons                                              |
| -------------------------------- | ------------------------------------------ | ------------------------------------------------- |
| Copy `.env.supabase` → `.env`    | Simple; all tools pick it up automatically | Overwrites local Docker config                    |
| Keep only `.env.supabase`        | Local `.env` untouched                     | Must `source` before scripts, or rely on fallback |
| `source .env.supabase` per shell | Explicit                                   | Easy to forget in a new terminal                  |

**Use `#` comments only** in env files. Do **not** use Python `"""` docstrings - shell `source` will fail.

```bash
cp nomikos/backend/core/.env.supabase.example nomikos/backend/core/.env.supabase
# edit credentials (never commit .env.supabase)
```

---

## Storage

### Bucket setup

1. **Storage** → New bucket → `document-media`
2. **Private** (no public access)
3. Backend uses **secret key** - no Storage policies needed for v1

### `STORAGE_BACKEND`

| Value             | Pros                                   | Cons                               |
| ----------------- | -------------------------------------- | ---------------------------------- |
| `local` (default) | Fast; no network; works offline        | Not shared across machines         |
| `supabase`        | Shared test DB + images; no local disk | Upload latency; bucket + key setup |

### WebP page images

All new uploads and seeds are normalized to **WebP**:

| Setting                                               | Default | Pros              | Cons                |
| ----------------------------------------------------- | ------- | ----------------- | ------------------- |
| `MEDIA_WEBP_LOSSLESS=true`                            | on      | Best OCR fidelity | Larger than lossy   |
| `MEDIA_WEBP_LOSSLESS=false` + `MEDIA_WEBP_QUALITY=95` | -       | Smaller files     | Slight quality loss |

Keys look like: `parts/<uuid>/<stem>.webp`

### Image serving

| Approach                    | Current                                      | Pros                               | Cons                    |
| --------------------------- | -------------------------------------------- | ---------------------------------- | ----------------------- |
| **API proxy** (implemented) | API reads bytes from Storage → HTTP response | Same auth as today; private bucket | More API bandwidth      |
| Signed URLs (future)        | Browser fetches Storage directly             | Offloads bandwidth                 | TTL + policy complexity |

---

## Auth

|           | App JWT (`JWT_SECRET`)                   | Supabase Auth                        |
| --------- | ---------------------------------------- | ------------------------------------ |
| **Used?** | Yes                                      | No                                   |
| **Pros**  | Full control; same code local + Supabase | Built-in OAuth, magic links          |
| **Cons**  | You manage secrets + rotation            | Second auth system; migration effort |

`JWT_SECRET` must be set in `.env.supabase` (≥32 bytes). It is **not** the Supabase secret key.

---

## Inference: local agent vs hosted agent

One path, two hosts. Every job is queued on the platform and claimed by an
**inference agent**; the only difference is which machine that agent runs on and
which credential it presents.

```text
Browser → API creates job → agent claims it → runs the model → job callback
             │                   │
             │                   ├─ researcher's machine, device token   → local
             │                   └─ persistent host, service credential  → cloud
             └─ execution target fixed here, from host preference + capacity
```

|                     | Local (researcher's machine)                            | Cloud (hosted agent)                                      |
| ------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| **Started by**      | The researcher: `nomikos pair`, then `nomikos run`     | An operator, as a supervised process                        |
| **Credential**      | Device token in `~/.nomikos/device.json`                | **Service credential** in `NOMIKOS_SERVICE_TOKEN`          |
| **Frontend config** | None - the editor reads **capacity** from the API         | None                                                        |
| **Backend env**     | `DEVICE_PAIRING_ENABLED`, `DEVICE_TOKEN_HMAC_SECRET`     | `INFERENCE_WORKER_SERVICE_TOKEN`                            |
| **Pros**            | No hosted inference cost; weights stay warm on one machine | Works with nothing installed; always available              |
| **Cons**            | Runs only while the researcher's terminal does            | Needs a persistent host                                     |

Which one a job gets is fixed at submission from the account's **host
preference** and **capacity**, and announced on the job. There is no per-job
toggle in the editor.

Neither agent holds `INFERENCE_WEBHOOK_SECRET`: an agent's job callback is
authorized by the same credential it claimed the page with, and narrowed to the
page it is holding. The secret still guards the platform's own webhook path, so
it stays configured on the API.

**Typical Supabase test setup:** Supabase DB + Storage, API local, agent on the
same machine pointed at `NOMIKOS_API_URL=http://localhost:8000`.

Nothing in this stack publishes an inference port. Compose runs no inference
container, and the agent listens on nothing.

---

## Runtime fixes (Supabase-specific)

### `sslmode` vs `ssl` (asyncpg)

|                 | psycopg2 (sync, Alembic) | asyncpg (`DATABASE_URL`) |
| --------------- | ------------------------ | ------------------------ |
| SSL query param | `?sslmode=require`       | `?ssl=require`           |

`infrastructure/db.py` rewrites `sslmode=` → `ssl=` for the async engine.

### PgBouncer prepared statements

**Error:** `DuplicatePreparedStatementError` / `prepared statement already exists`

**Cause:** Transaction pooler (`:6543`) does not support asyncpg prepared statement cache.

**Fix (automatic):** When URL contains `pooler.supabase.com` or `:6543`, async engine sets `connect_args={"statement_cache_size": 0}`.

|          | Statement cache on                         | Statement cache off        |
| -------- | ------------------------------------------ | -------------------------- |
| **Pros** | Faster repeated queries on direct Postgres | Works with Supabase pooler |
| **Cons** | Breaks on transaction pooler               | Slight per-query overhead  |

---

## Step-by-step workflow

### 1. One-time Supabase project

- [ ] Create project; save **database password**
- [ ] Disable auto-expose tables + auto RLS (see above)
- [ ] Create private bucket `document-media`
- [ ] Copy **secret key** and connection strings

### 2. Configure env

```bash
cp nomikos/backend/core/.env.supabase.example nomikos/backend/core/.env.supabase
```

Fill `.env.supabase` from the provider secret store with the API and migrator
DB URLs, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `JWT_SECRET`, and
`STORAGE_BACKEND=supabase`. There is no separate inference environment file:
an inference agent needs no database access, and a hosted agent reads its
service credential from `NOMIKOS_SERVICE_TOKEN` in its own process
environment.

### 3. Migrate

```bash
# The schema migration creates the service groups when the operator permits it.
./scripts/platform/migrate_supabase.sh
```

For the disposable pre-production project only, reset the application schema
and rerun the consolidated migrations with an explicit guard:

The guard is deliberately not satisfiable from your shell. `SUPABASE_NON_PRODUCTION=true`
must appear inside the environment file itself, which is parsed before anything is
merged into the process environment, so an `export` in your profile will not do it, and
neither will an `ENVIRONMENT` of `production` in that file. Add to `.env.supabase`:

```dotenv
SUPABASE_NON_PRODUCTION=true
ENVIRONMENT=staging
```

Then run it. Interactively, the script prints the resolved target (the Supabase
project ref parsed from `MIGRATOR_DATABASE_URL`, cross-checked against `SUPABASE_URL`)
and requires you to type it back:

```bash
./scripts/platform/reset_supabase_nonprod.sh
```

Unattended, name the same target explicitly. A fixed literal such as `RESET` is
rejected, because a confirmation that is identical for every project confirms nothing:

```bash
CONFIRM_SUPABASE_RESET=<project-ref> ./scripts/platform/reset_supabase_nonprod.sh --yes
```

In CI, set `SUPABASE_PRODUCTION_PROJECT_REFS` to a comma-separated denylist of refs
that must never be reset, whatever the environment file claims.

This drops only the application tables, enums, Alembic history, and obsolete
RLS helper functions before applying `001_initial_schema` and
`002_service_roles`. It does not delete Supabase Storage objects; clear the
disposable `document-media` bucket separately if required. Never set the
confirmation variables for a production project.

### 4. Seed

```bash
uv run python scripts/platform/seed_dev_user.py
uv run python scripts/platform/seed_dev_inference.py
uv run python scripts/platform/seed_dev_annotated_data.py   # optional corpus
```

### 5. Run API

```bash
cd nomikos
PYTHONPATH=. uvicorn backend.core.app:create_app --factory --reload --port 8000
```

### 5b. Run full stack with Docker (recommended)

From the **repository root**, with `.env.supabase` configured:

```bash
# First time (or after code changes)
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml up --build

# Later starts
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml up

# Background
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml up -d --build
```

| URL                   | Service      |
| --------------------- | ------------ |
| http://localhost:5173 | Frontend     |
| http://localhost:8000 | Platform API |

This profile **does not start local Postgres** (`db` is disabled). Apply
Alembic from the operator/migrator host first; the Compose API only runs the
idempotent development seed.

```bash
# Stop
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml down

# Logs
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml logs -f api frontend

# Rebuild one service
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml up --build api
```

The **inference agent** is not in Compose - run it on the host if you want jobs
to execute:

```bash
NOMIKOS_API_URL=http://localhost:8000 uv run --group inference python -m nomikos_inference.cli pair
NOMIKOS_API_URL=http://localhost:8000 uv run --group inference python -m nomikos_inference.cli run
```

It reaches the API the same way your browser does, so it works from the host
against the Compose API without any published port of its own. The page editor
never contacts it: what the editor shows about this machine is **capacity** read
from the API.

### 6. Run frontend (without Docker)

```bash
# nomikos/frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Dev login after seed: `dev@example.com` / `dev-pass-123`

---

## Environment reference

| Variable                    | Required (Supabase) | Purpose                                            |
| --------------------------- | ------------------- | -------------------------------------------------- |
| `MIGRATOR_DATABASE_URL`     | Yes                 | Alembic operator/migrator - direct Postgres        |
| `DATABASE_URL`              | Yes                 | Async SQLAlchemy (`+asyncpg`) API/worker principal |
| `SYNC_DATABASE_URL`         | Yes                 | Matching API/worker principal sync connection      |
| `STORAGE_BACKEND`           | Yes                 | `supabase`                                         |
| `SUPABASE_URL`              | Yes                 | Project API URL                                    |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes                 | Storage (secret key)                               |
| `SUPABASE_STORAGE_BUCKET`   | Yes                 | Default `document-media`                           |
| `JWT_SECRET`                | Yes                 | App auth (not Supabase)                            |
| `MEDIA_WEBP_LOSSLESS`       | No                  | Default `true`                                     |
| `INFERENCE_WEBHOOK_SECRET`  | Yes in production   | Authenticates the platform's own webhook callback path - not an agent's |
| `DEVICE_TOKEN_HMAC_SECRET`  | Yes when pairing is on | Keys every device token; must differ from `JWT_SECRET`, or a JWT rotation unpairs every machine |
| `INFERENCE_WORKER_SERVICE_TOKEN` | Only for a hosted agent | The **service credential** that claims `cloud` work |

---

## Local vs Supabase summary

|              | Local dev                            | Supabase test deploy      |
| ------------ | ------------------------------------ | ------------------------- |
| Database     | Docker `kalamos` @ `:5433`           | Supabase `postgres`       |
| Migrations   | `alembic upgrade head`               | Same Alembic → direct URL |
| Page images  | `MEDIA_ROOT` filesystem              | Storage bucket (WebP)     |
| Auth         | App JWT                              | App JWT                   |
| Inference    | Agent on the host                    | Agent on the host typical |
| Cost / setup | Free, offline                        | Hosted; needs network     |

---

## Troubleshooting

| Symptom                                               | Cause                                      | Fix                                                                      |
| ----------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------ |
| `No such file or directory` on migrate                | Python `"""` in `.env.supabase`            | Use `#` comments only                                                    |
| `JWT_SECRET` missing                                  | Only `.env.supabase` exists, old code path | Update repo; or copy to `.env`                                           |
| `connect() got unexpected keyword argument 'sslmode'` | asyncpg URL                                | Use `ssl=` or let `db.py` rewrite                                        |
| `DuplicatePreparedStatementError`                     | Pooler + asyncpg                           | Fixed in `db.py`; restart process                                        |
| `role nomikos_app cannot be dropped`                 | Legacy pre-squash role                     | Remove it during a non-production reset or through the provider operator |
| Upload 401/403 to Storage                             | Wrong key or missing bucket                | Secret key + private `document-media`                                    |
| Connection refused on `:5433`                         | Docker not running                         | `docker compose -f infrastructure/docker-compose.yml up db -d`                                                |
| Integration tests hang                                | Stale DB advisory locks                    | Stop API; terminate idle sessions                                        |

---

## Related docs

- [Supabase learnings (pitfalls + connection URLs)](../guides/learnings.md#supabase-hosted-postgres--storage)
- [Self-hosting and local inference](../../README.md#self-hosting-and-local-inference)
- [Local development guide](../guides/local-development.md)
- [Infrastructure README](../../nomikos/infrastructure/README.md)
