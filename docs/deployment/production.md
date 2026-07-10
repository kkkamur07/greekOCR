# Production deployment

Host Nomicous at **nomicous.com** with four public surfaces and Supabase as the shared database + storage backend.

Architecture overview: [root README](../../README.md#production-hosting). Serverless constraints and pitfalls: [learnings — Vercel](../guides/learnings.md#serverless-api-vercel).

| Domain | Vercel project | Root directory | Role |
|--------|----------------|----------------|------|
| [nomicous.com](https://nomicous.com) | `nomicous-landing` | `landing/` | Static marketing site |
| [app.nomicous.com](https://app.nomicous.com) | `nomicous-app` | `nomicous/frontend/` | Next.js App Router client |
| [api.nomicous.com](https://api.nomicous.com) | `nomicous-api` | `deploy/platform/` | FastAPI platform API |
| [inference.nomicous.com](https://inference.nomicous.com) | **Not Vercel** — optional | Docker image | Optional cloud ML inference API + worker |

Supabase setup: [`supabase.md`](supabase.md).
Local development: [`../guides/local-development.md`](../guides/local-development.md).

---

## Architecture

```text
Browser
  ├─ nomicous.com          → Vercel (static landing)
  ├─ app.nomicous.com      → Vercel (Next.js App Router)
  │     ├─ REST + JWT      → api.nomicous.com (Vercel serverless FastAPI)
  │     └─ local OCR       → 127.0.0.1:8001 Inference Helper (user machine)
  │
  ├─ api.nomicous.com      → Supabase Postgres + Storage
  └─ optional inference.nomicous.com → Supabase Postgres (inference_jobs table)

Background (persistent compute, not serverless):
  platform-worker   → claims platform jobs, submits to inference API
  inference-worker  → runs OCR/segment models, callbacks to api
```

### Why local inference is the default

The local Inference Helper bundles the **PyTorch Calamari runtime** and **Kraken** and
runs jobs for up to **30 minutes**. Hub weights are resolved lazily into the
runtime cache; the default Kraken BLLA asset comes from the installed `kraken`
package. Vercel serverless functions have strict size limits and short execution
timeouts, so researchers run inference on their own machines through the
loopback-only helper; the hosted platform persists only the result.

Cloud inference remains optional. If it is enabled later, deploy the existing
[`inference/Dockerfile`](../inference/Dockerfile) and the platform worker on a
persistent host (see [`deploy/inference/README.md`](../../deploy/inference/README.md)).

Local inference via the **Inference Helper** (DMG installer) remains the primary path for researchers who want on-device OCR; cloud inference is optional.

---

## 1. Supabase (once per environment)

1. Create a Supabase project (database name `postgres`).
2. Create a **private** Storage bucket `document-media`.
3. Provision database service roles and run migrations:

```bash
cp nomicous/backend/core/.env.supabase.example nomicous/backend/core/.env.supabase
# Store credentials only in the provider's secret manager.
# Follow docs/deployment/database-roles.md before this first migration.
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
| Framework | Next.js |
| Build Command | `npm run build` |
| Output Directory | *(default)* |
| Domain | `app.nomicous.com` |

Environment variables (Production):

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.nomicous.com
NEXT_PUBLIC_CSRF_COOKIE_NAME=greekocr-csrf
NEXT_PUBLIC_INFERENCE_HELPER_URL=http://localhost:8001
NEXT_PUBLIC_ENABLE_TEST_JOBS=false
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
| Function region | `fra1` (Frankfurt, Europe) |

Environment variables: copy from [`nomicous/backend/core/.env.production.example`](../nomicous/backend/core/.env.production.example).

**Critical serverless settings:**

| Variable | Production value | Why |
|----------|------------------|-----|
| `JOB_WORKER_ENABLED` | `false` | Worker runs on persistent host |
| `JOB_SSE_NOTIFICATIONS_ENABLED` | `false` | NOTIFY listener needs long-lived process |
| `BEHIND_PROXY` | `false` (current Vercel deployment) | Forwarded headers are not trusted without a fixed proxy allowlist |
| `FORWARDED_ALLOW_IPS` | Unset (current Vercel deployment) | Set explicit IP/CIDRs before enabling `BEHIND_PROXY`; never `*` |
| `CORS_ORIGINS` | `https://app.nomicous.com` | Browser origin |
| `STORAGE_BACKEND` | `supabase` | No local filesystem on Vercel |

Job progress in the browser falls back to **HTTP polling** when SSE is unavailable (already implemented in the frontend).

`FORWARDED_ALLOW_IPS` accepts only explicit proxy IPs or CIDRs. Do not trust
forwarded headers on Vercel unless the request reaches the function from a
stable, allowlisted proxy address; otherwise set `BEHIND_PROXY=false` and rate
limits use the direct connection peer.

Config: [`deploy/platform/vercel.json`](../../deploy/platform/vercel.json).

The `fra1` setting applies to the serverless platform API only. The landing
page and frontend SPA remain globally edge-served by Vercel. Validate API p95
latency against the Supabase project region after deployment; rollback by
removing `regions` from [`deploy/platform/vercel.json`](../../deploy/platform/vercel.json).

Vercel-specific Python runtime, bundle-size, and dependency notes:
[`vercel-platform-api.md`](vercel-platform-api.md).

---

## 3. Optional cloud inference + platform worker (persistent host)

Skip this section for the default local-helper deployment. If you later enable
cloud jobs, see [`deploy/inference/README.md`](../../deploy/inference/README.md).

Minimum services:

| Service | Command | Port |
|---------|---------|------|
| `inference-api` | `uvicorn inference.api.main:app --host 0.0.0.0 --port 8001` | 8001 |
| `inference-worker` | `python -m inference.jobs.worker` | — |
| `platform-worker` | `python -m backend.jobs.worker_main` | — |

Cloud inference is currently disabled. If it is enabled later, set
`CLOUD_INFERENCE_ENABLED=true` and point `INFERENCE_URL` on the platform API to
`https://inference.nomicous.com`. Do not set the platform API's production
`INFERENCE_URL` to `localhost:8001`; that address belongs to the browser-side
local helper and is configured with `NEXT_PUBLIC_INFERENCE_HELPER_URL`.
Set `INFERENCE_CALLBACK_URL` on inference to `https://api.nomicous.com/internal/inference/job-complete`.
Use the distinct API, platform-worker, and inference-worker database principals
from [`database-roles.md`](database-roles.md); do not give these containers the
Supabase operator/migration URI.

---

## 4. DNS checklist

| Record | Type | Target |
|--------|------|--------|
| `nomicous.com` | A / CNAME | Vercel landing project |
| `www` | CNAME | Vercel (redirect to apex) |
| `app` | CNAME | Vercel app project |
| `api` | CNAME | Vercel API project |
| `inference` | CNAME | Railway / Fly inference host (optional cloud jobs) |

---

## 5. Inference Helper (local OCR)

Ship the macOS DMG from GitHub Releases. The installer configures the loopback-only
helper for `https://app.nomicous.com` at runtime; it does not embed a browser secret.
For another stable app origin, set `HELPER_CORS_ORIGINS` while installing as described
in [`packaging/helper/README.md`](../../packaging/helper/README.md). Do not use wildcard
origins or preview domains.

---

## 6. Pre-launch checklist

Record the required CI, secret-rotation, migration, smoke-test, metric, and
rollback evidence in the [production release record](release-evidence.md).

### Secrets and history response

Production credentials are configured only in the provider secret store for the
service that consumes them. Committed `*.env.example` files are documentation;
filled `.env*` files remain local and ignored.

If a secret scanner or Git-history review identifies a possible exposure:

1. Inventory affected paths and consumers without copying the value into an
   issue, log, or chat.
2. Create a scoped replacement in the relevant provider secret store.
3. Deploy and verify every consumer with the replacement.
4. Revoke the old credential.
5. Record the rotation date, owner, affected service, and verification result;
   coordinate history remediation separately when the value was tracked.

### Code and security gates

- [ ] Frontend typecheck, lint, build, and tests pass in CI
- [ ] First-party Python tests and Ruff checks pass in CI
- [ ] OpenAPI/generated-client drift check passes
- [ ] Dependency, secret, and container vulnerability scans pass
- [ ] No production credentials exist in the working tree, Git history, or build artifacts
- [ ] `JWT_SECRET`, `INFERENCE_WEBHOOK_SECRET`, and `INFERENCE_SERVICE_SECRET` are unique per environment
- [ ] Upload, inference, callback, authorization, and cross-user isolation tests pass
- [ ] Model checkpoints use a safe loading path, pinned revision, and verified manifest/hash

### Infrastructure and application checks

- [ ] Supabase migrations applied (`alembic upgrade head`)
- [ ] Service-role bootstrap completed and each runtime has only its own DB URL
- [ ] `CORS_ORIGINS` includes only production app origins
- [ ] `ENABLE_TEST_JOB_ROUTES=false`
- [ ] Platform worker + inference worker running and healthy
- [ ] Vercel bundle contains no model weights, local media, `.env` files, or training artifacts
- [ ] Docker images build, import, health-check, and run as non-root where applicable
- [ ] Release checksums, SBOMs, vulnerability scans, and provenance attestations are available
- [ ] Raw/encoded upload limits, decoded pixel limits, job limits, timeouts, and rate limits are enabled
- [ ] Upload a test page image → appears in Supabase Storage as WebP
- [ ] Login/register on `app.nomicous.com`
- [ ] Cloud segment/transcribe job completes end-to-end
- [ ] Local helper install modal links to GitHub release

### Accessibility and user-flow checks

- [ ] Critical flows work with keyboard navigation only
- [ ] Modal focus is trapped and restored correctly
- [ ] Form errors are descriptive and associated with their fields
- [ ] Public canvas controls have accessible names and keyboard behavior
- [ ] No critical axe or Lighthouse accessibility findings remain

### Deployment and observation

- [ ] Record API p50/p95/p99 latency, error rate, request volume, and `/health` response before deployment
- [ ] Deploy a preview/staging build and run the smoke checks before production
- [ ] Deploy `api.nomicous.com` with function region `fra1`
- [ ] Confirm Vercel logs show the expected region and contain no secrets or submitted payloads
- [ ] During the first hour, verify health, login, upload, job submission, polling, storage, and export
- [ ] Compare European p95 latency and error rate with the pre-deployment baseline
- [ ] Keep the prior deployment available and record the rollback operator

Advance only when error rate stays within 10% of baseline and p95 latency stays
within 20% of baseline. Hold and investigate if either metric is further above
baseline. Roll back for an error rate above 2× baseline, p95 latency above 50%
baseline, a security issue, data-integrity problem, cross-user exposure, or a
broken critical flow.

### Rollback

1. Restore the last known-good Vercel deployment.
2. If the issue is specific to regional placement, remove `regions` from
   [`deploy/platform/vercel.json`](../../deploy/platform/vercel.json) and
   redeploy.
3. Re-run `/health` and the critical-flow smoke checks.
4. Compare error rate, latency, job completion, and storage writes with the
   baseline.
5. Record the incident, decision, timestamps, and any required follow-up in
   the relevant issue or [`docs/guides/learnings.md`](../guides/learnings.md).

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
