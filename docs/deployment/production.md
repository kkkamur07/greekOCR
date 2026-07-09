# Production deployment

Host Nomicous at **nomicous.com** with four public surfaces and Supabase as the shared database + storage backend.

| Domain | Vercel project | Root directory | Role |
|--------|----------------|----------------|------|
| [nomicous.com](https://nomicous.com) | `nomicous-landing` | `landing/` | Static marketing site |
| [app.nomicous.com](https://app.nomicous.com) | `nomicous-app` | `nomicous/frontend/` | React SPA (Vite) |
| [api.nomicous.com](https://api.nomicous.com) | `nomicous-api` | `deploy/platform/` | FastAPI platform API |
| [inference.nomicous.com](https://inference.nomicous.com) | **Not Vercel** — see below | Docker image | ML inference API + worker |

Supabase setup: [`supabase.md`](supabase.md).  
Local development: [`../guides/local-development.md`](../guides/local-development.md).

---

## Architecture

```text
Browser
  ├─ nomicous.com          → Vercel (static landing)
  ├─ app.nomicous.com      → Vercel (React SPA)
  │     ├─ REST + JWT      → api.nomicous.com (Vercel serverless FastAPI)
  │     └─ optional        → localhost:8001 Inference Helper (user machine)
  │
  ├─ api.nomicous.com      → Supabase Postgres + Storage
  └─ inference.nomicous.com → Supabase Postgres (inference_jobs table)

Background (persistent compute, not serverless):
  platform-worker   → claims platform jobs, submits to inference API
  inference-worker  → runs OCR/segment models, callbacks to api
```

### Why inference is not on Vercel

The inference service bundles **PyTorch, Kraken, and model weights** and runs jobs for up to **30 minutes**. Vercel serverless functions have strict size limits and short execution timeouts. Deploy inference with the existing [`inference/Dockerfile`](../inference/Dockerfile) on **Railway**, **Fly.io**, or similar.

The **platform job worker** also needs a long-running process. Run it as a second service on the same host as inference (see [`deploy/inference/README.md`](../../deploy/inference/README.md)).

Local inference via the **Inference Helper** (DMG installer) remains the primary path for researchers who want on-device OCR; cloud inference is optional.

---

## 1. Supabase (once per environment)

1. Create a Supabase project (database name `postgres`).
2. Create a **private** Storage bucket `document-media`.
3. Run migrations:

```bash
cp nomicous/backend/core/.env.supabase.example nomicous/backend/core/.env.supabase
# fill in credentials
./scripts/platform/migrate_supabase.sh
```

4. Seed production data if needed (admin user, model catalog) — do **not** run dev seed scripts in production.

---

## 2. Vercel projects

Connect the same GitHub repo to **three** Vercel projects. Set the **Root Directory** per project (Project Settings → General).

### Landing (`nomicous-landing`)

| Setting | Value |
|---------|-------|
| Root Directory | `landing` |
| Framework | Other (static) |
| Build Command | *(empty)* |
| Output Directory | `.` |
| Domain | `nomicous.com`, `www.nomicous.com` → redirect to apex |

Config: [`landing/vercel.json`](../landing/vercel.json).

### App (`nomicous-app`)

| Setting | Value |
|---------|-------|
| Root Directory | `nomicous/frontend` |
| Framework | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Domain | `app.nomicous.com` |

Environment variables (Production):

```bash
VITE_API_BASE_URL=https://api.nomicous.com
```

Template: [`nomicous/frontend/.env.production.example`](../nomicous/frontend/.env.production.example).

Config: [`nomicous/frontend/vercel.json`](../nomicous/frontend/vercel.json).

### Platform API (`nomicous-api`)

| Setting | Value |
|---------|-------|
| Root Directory | `deploy/platform` |
| Install Command | *(empty / default)* |
| Build Command | `bash build.sh` |
| Output Directory | `.` |
| Domain | `api.nomicous.com` |

Environment variables: copy from [`nomicous/backend/core/.env.production.example`](../nomicous/backend/core/.env.production.example).

**Critical serverless settings:**

| Variable | Production value | Why |
|----------|------------------|-----|
| `JOB_WORKER_ENABLED` | `false` | Worker runs on persistent host |
| `JOB_SSE_NOTIFICATIONS_ENABLED` | `false` | NOTIFY listener needs long-lived process |
| `BEHIND_PROXY` | `true` | Vercel terminates TLS |
| `FORWARDED_ALLOW_IPS` | `*` | Trust Vercel proxy headers |
| `CORS_ORIGINS` | `https://app.nomicous.com` | Browser origin |
| `STORAGE_BACKEND` | `supabase` | No local filesystem on Vercel |

Job progress in the browser falls back to **HTTP polling** when SSE is unavailable (already implemented in the frontend).

Config: [`deploy/platform/vercel.json`](../../deploy/platform/vercel.json).

---

## 3. Inference + platform worker (persistent host)

See [`deploy/inference/README.md`](../../deploy/inference/README.md).

Minimum services:

| Service | Command | Port |
|---------|---------|------|
| `inference-api` | `uvicorn inference.api.main:app --host 0.0.0.0 --port 8001` | 8001 |
| `inference-worker` | `python -m inference.jobs.worker` | — |
| `platform-worker` | `python -m backend.jobs.worker_main` | — |

Point `INFERENCE_URL` on the platform API to `https://inference.nomicous.com`.
Set `INFERENCE_CALLBACK_URL` on inference to `https://api.nomicous.com/internal/inference/job-complete`.

---

## 4. DNS checklist

| Record | Type | Target |
|--------|------|--------|
| `nomicous.com` | A / CNAME | Vercel landing project |
| `www` | CNAME | Vercel (redirect to apex) |
| `app` | CNAME | Vercel app project |
| `api` | CNAME | Vercel API project |
| `inference` | CNAME | Railway / Fly inference host |

---

## 5. Inference Helper (local OCR)

Ship the macOS DMG from GitHub Releases. Set `HELPER_CORS_ORIGINS=https://app.nomicous.com` when building production helper packages (see [`packaging/helper/README.md`](../packaging/helper/README.md)).

---

## 6. Pre-launch checklist

- [ ] Supabase migrations applied (`alembic upgrade head`)
- [ ] `JWT_SECRET`, `INFERENCE_WEBHOOK_SECRET`, `INFERENCE_SERVICE_SECRET` are unique per environment
- [ ] `CORS_ORIGINS` includes only production app origins
- [ ] `ENABLE_TEST_JOB_ROUTES=false`
- [ ] Platform worker + inference worker running and healthy
- [ ] Upload a test page image → appears in Supabase Storage as WebP
- [ ] Login/register on `app.nomicous.com`
- [ ] Cloud segment/transcribe job completes end-to-end
- [ ] Local helper install modal links to GitHub release

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| CORS error from app | `CORS_ORIGINS` missing `https://app.nomicous.com` |
| 401 on all API calls | Wrong `JWT_SECRET` or clock skew |
| Jobs stuck in `pending` | Platform worker not running (`JOB_WORKER_ENABLED` must be `false` on API, worker elsewhere) |
| Jobs stuck in `waiting` | Inference worker not running |
| `DuplicatePreparedStatementError` | Using pooler URL without `statement_cache_size=0` — already handled in `infrastructure/db.py` |
| Media 404 | `STORAGE_BACKEND=supabase` but bucket/key wrong |
| SSE never connects | Expected on Vercel; polling fallback should still complete jobs |
