# Learnings

Operational notes and frequent errors from building and running Kalamos / Nomikos. For step-by-step setup, see [local development](local-development.md) and [deployment/supabase.md](../deployment/supabase.md).

---

## Supabase (hosted Postgres + Storage)

We use **Supabase** for **shared test/staging and production database hosting**, not as a full BaaS. Local Docker Postgres remains the default for day-to-day dev.

| Layer | Supabase product | How we use it |
|-------|------------------|---------------|
| Database | Managed Postgres | SQLAlchemy + **Alembic** migrations (same history local and remote) |
| Page images | Storage (`document-media` bucket) | `MediaStore` backend (`STORAGE_BACKEND=local` \| `supabase`) |
| Auth / Data API | **Not used** | App JWT + FastAPI authorization unchanged |

**Alembic** is the schema source of truth. We do **not** use the Supabase CLI for migrations.

### What we deliberately skipped

| Supabase feature | Why not |
|------------------|---------|
| Supabase Auth | Existing JWT + user tables |
| PostgREST / Data API | Backend talks SQL directly via SQLAlchemy |
| Row Level Security (RLS) | Disabled; authorization remains in FastAPI |

### Connection URLs

Supabase exposes multiple Postgres endpoints. We use:

| URL env var | Endpoint | Used for |
|-------------|----------|----------|
| `MIGRATOR_DATABASE_URL` | Direct (session mode) | `alembic upgrade` |
| `DATABASE_URL` | Pooler (transaction mode) | Async API runtime (`asyncpg`) |
| `SYNC_DATABASE_URL` | Pooler or direct | Sync workers, NOTIFY listeners |

### Frequent errors (Supabase)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `prepared statement "..." already exists` with pooler | Transaction pooler + asyncpg statement cache | Set `statement_cache_size=0` on async engine (see `infrastructure/db.py`) |
| Alembic fails on password with `%` or `@` | URL special characters | Percent-encode password in connection URL; Alembic `env.py` doubles `%` for ConfigParser |
| Migration permission errors on new tables | Supabase default privileges | Run migrations with direct `MIGRATOR_DATABASE_URL`; verify `002_service_roles` grants |
| Images 404 in Supabase profile | `STORAGE_BACKEND` still `local` or missing bucket key | Set `STORAGE_BACKEND=supabase`, `SUPABASE_SERVICE_ROLE_KEY`, bucket name in `.env.supabase` |
| SSL required | Hosted Postgres | `sslmode=require` on remote URLs |
| Schema drift between dev and staging | Migrations not run on Supabase | `./scripts/platform/migrate_supabase.sh` after pulling Alembic revisions |

### Quick start (Supabase profile)

```bash
cp nomikos/backend/core/.env.supabase.example nomikos/backend/core/.env.supabase
# fill DATABASE_URL, MIGRATOR_DATABASE_URL, SUPABASE_* keys
./scripts/platform/migrate_supabase.sh
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml up --build
```

Full guide: [`docs/deployment/supabase.md`](../deployment/supabase.md).

---

## Serverless API (Vercel)

Production ships the platform API as a **Vercel Python serverless function** (`infrastructure/platform/` → `api.nomikos.app`). This works for REST + JWT + Supabase, but **not** for background workers or SSE listeners.

### What must run elsewhere

| Capability | Why not serverless | Where it runs |
|------------|-------------------|---------------|
| Platform job worker (`claim` → submit to inference) | Needs continuous polling / claiming | `platform-worker` on persistent Docker host |
| Model execution (ONNX Runtime) | Bundle size, 30+ min jobs | The **inference agent** — a laptop, or a hosted worker claiming with a service credential. There is no inference service to deploy (ADR 0002/0003) |
| Job status SSE (`NOTIFY` → browser stream) | `LISTEN` is a long-lived DB connection | Disabled on Vercel; enabled in Docker Compose / future all-Docker deploy |

### Required env on Vercel

| Variable | Value | Reason |
|----------|-------|--------|
| `JOB_WORKER_ENABLED` | `false` | Worker runs as `python -m backend.jobs.worker_main` on Docker host |
| `JOB_SSE_NOTIFICATIONS_ENABLED` | `false` | No persistent NOTIFY listener in serverless |
| `BEHIND_PROXY` | `false` unless a fixed proxy range is available | Enables forwarded headers only from trusted peers |
| `FORWARDED_ALLOW_IPS` | Explicit proxy IP/CIDR list | Required with `BEHIND_PROXY=true`; never use `*` |
| `STORAGE_BACKEND` | `supabase` | No writable local disk |

Platform API functions are pinned to **`fra1` (Frankfurt)** in `infrastructure/platform/vercel.json` for European latency. Landing and SPA stay globally edge-served. Rollback: remove `regions` from `vercel.json` and redeploy.

Frontend falls back to **HTTP polling** when SSE is unavailable - verified in production.

### Future: all-Docker on our server

We plan a **persistent Docker deployment** (API + workers on one host) where `JOB_WORKER_ENABLED=true`, `JOB_SSE_NOTIFICATIONS_ENABLED=true`, and SSE replaces polling. The Vercel path remains valid for web surfaces; the env flags are the switch.

### Frequent errors (serverless)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Jobs stuck in `pending` | `platform-worker` not running | Deploy worker on Docker; keep `JOB_WORKER_ENABLED=false` on Vercel API |
| Jobs stuck in `waiting` | No inference agent finished the page it claimed | Check the agent is running and its callback URL is reachable |
| SSE never connects on production | Expected on Vercel | Polling fallback should still complete jobs |
| CORS errors from `app.nomikos.app` | Missing origin | Set `CORS_ORIGINS=https://app.nomikos.app` on API |
| Wrong scheme / redirect loops | Proxy headers | Set `BEHIND_PROXY=true` only with an explicit `FORWARDED_ALLOW_IPS` proxy IP/CIDR list |
| Media 404 | Local storage on serverless | `STORAGE_BACKEND=supabase` + bucket configured |
| Function too large / timeout | ML code in API bundle | Inference stays in separate Docker image only |

### Production deployment incident: July 2026

The API build completed successfully, but the first production requests returned
`FUNCTION_INVOCATION_FAILED`. The failure happened while importing the FastAPI
application: production settings were intentionally stricter than the old
Vercel environment. The runtime never reached a route handler.

The fixes were configuration changes, not application workarounds:

| Startup error | Why it failed | Production fix |
|--------------|---------------|----------------|
| `FORWARDED_ALLOW_IPS=*` | Wildcard forwarded-header trust permits spoofed client IPs | Remove the variable |
| `BEHIND_PROXY=true` without a CIDR | Forwarded headers were enabled without a known trusted proxy range | Set `BEHIND_PROXY=false` |
| Placeholder/missing inference secret | Production callbacks require a real shared secret | Store `INFERENCE_WEBHOOK_SECRET` as an encrypted Vercel variable |
| Only `VITE_*` frontend variables | The frontend migrated from Vite to Next.js | Use `NEXT_PUBLIC_*` names |

The API is deployed at **`https://api.nomikos.app`** and pinned to Vercel's
Frankfurt region (`fra1`). The landing page and frontend remain globally served.
After configuration changes, verify the API before testing application flows:

```bash
curl -sS https://api.nomikos.app/health
# {"status":"ok","database":"ok"}
```

### Local versus cloud inference

Local inference runs as a CLI the user installs, not as a service the browser
dials (ADR 0002). The agent connects **out** to the platform and claims work;
nothing listens on the laptop, so no browser's local-network policy, no
loopback CORS contract and no mixed-content rule is on the path. The frontend's
old probe order, the loopback `connect-src` entries and the
`Access-Control-Allow-Private-Network` preflight were all deleted by #60;
`tests/nomikos/unit/test_deployment_hardening.py` now asserts the CSP grants
no loopback origin.

The Vercel API never dials out to an inference host, which is what removed a
whole class of this mistake: `localhost` inside a Vercel function means the
ephemeral function container, not the researcher's computer. Inference is
inbound now — an agent claims from the API (ADR 0003). Keep cloud inference
disabled until a persistent agent host is available.

Runbook: [`docs/deployment/production.md`](../deployment/production.md). Vercel Python notes: [`docs/deployment/vercel-platform-api.md`](../deployment/vercel-platform-api.md).

---

## Calamari training

The training entry points below are currently broken. The `uv` extras and
Hydra config paths they reference do not exist, so none of these commands run
as written. See `docs/final-code-review-2026-08-06.md` for the exact failures
before trusting this section.

Training and finetuning use the **vendored TensorFlow Calamari tree** under `src/model/calamari/`. Inference does **not** import that tree - it runs a separate ONNX Runtime graph under `nomikos_inference/architectures/calamari/` and loads `best.onnx` from Hugging Face Hub (`hf://`).

### Layout

```text
src/model/calamari/          # Canonical vendored Calamari (in git)
  calamari_ocr/              # Python package imported at training time
    ocr/dataset/             # Required - do not omit when syncing from upstream
    scripts/train.py
    ...

src/train/calamari/          # Hydra entry points (train.py, finetune.py)
configs/calamari_*.yaml      # Training presets
outputs/                     # Checkpoints (e.g. outputs/calamari-greek-bible/best.ckpt)

_support_repo/calamari/      # Optional legacy symlink for older scripts (gitignored)
```

Training scripts resolve vendored code directly from `src/model/calamari` (`src/train/calamari/train_utils.py`). You do **not** need `_support_repo` unless a one-off script still expects that path.

### Quick start

From the repository root:

```bash
uv sync --group train
PYTHONPATH=. python src/train/calamari/train.py
```

Finetune: `./src/train/calamari/finetune.sh`

Checkpoints land under `outputs/` (override via Hydra `output.root` in `configs/calamari_train.yaml`).

### Optional `_support_repo` symlink

```bash
mkdir -p _support_repo
ln -sfn ../src/model/calamari _support_repo/calamari
```

### Frequent errors (Calamari)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: Local Calamari source not found at .../src/model/calamari` | Incomplete clone | Ensure `src/model/calamari/calamari_ocr/scripts/train.py` exists |
| `ModuleNotFoundError: No module named 'calamari_ocr.ocr.dataset'` | Incomplete vendored tree | Sync full `calamari_ocr/ocr/dataset/` from the Calamari fork |
| `ModuleNotFoundError: No module named 'calamari_ocr'` with `_support_repo` in traceback | Legacy path expected | Symlink `_support_repo/calamari` → `src/model/calamari` |
| Training OOM / very slow on Apple Silicon | TensorFlow + emulation | Prefer Linux + GPU; reduce batch size |
| `Expected train/ and val/ images under ...` | Wrong pack layout | Pack needs `train/` and `val/` image folders |
| Checkpoint works in training but inference rejects `.ckpt` | Inference needs the ONNX graph | Export `best.onnx` and publish the graph to Hub |
| Empty OCR despite job **done** | Wrong model for script | Use a model matching the page script (e.g. Syriac model only for Syriac pages) |
| First Hub transcribe slow in Docker | Cold download + CPU | `PYTHONPATH=. python scripts/hf/fetch_model.py syriac-calamari-v1 --registry-tag stable` (Greek model not published yet) |

---

## Related docs

| Doc | Topic |
|-----|--------|
| [Root README](../../README.md) | Repo overview, Supabase summary, production hosting |
| [deployment/production.md](../deployment/production.md) | Vercel + Supabase + Docker runbook |
| [deployment/supabase.md](../deployment/supabase.md) | Supabase operational guide |
| [nomikos_inference/README.md](../../nomikos_inference/README.md) | Inference service, Hub cache |
