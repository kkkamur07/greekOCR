# Platform API - Vercel deploy

Serverless FastAPI bundle for **api.nomicous.com**. Full production runbook: [`docs/deployment/production.md`](../../docs/deployment/production.md).
Vercel-specific build and bundle notes: [`docs/deployment/vercel-platform-api.md`](../../docs/deployment/vercel-platform-api.md).

## What this folder is

Vercel's **Root Directory** for the `nomicous-api` project. At build time `build.sh` copies the platform backend and inference registry metadata into this tree; generated artifacts are gitignored.

| Path | Purpose |
|------|---------|
| `api/index.py` | ASGI entrypoint (`app` export) |
| `requirements.txt` | Platform API runtime dependencies |
| `build.sh` | Bundle backend sources into this tree |
| `vercel.json` | Rewrites, 60s function timeout |

## Local smoke test

```bash
cd deploy/platform
bash build.sh
PYTHONPATH=".:nomicous" python -c "from api.index import app; print(app.title)"
```

## Vercel project settings

| Setting | Value |
|---------|-------|
| Root Directory | `deploy/platform` |
| Framework | Other |
| Install Command | *(empty / default)* |
| Build Command | `bash build.sh` |
| Output Directory | `.` |
| Domain | `api.nomicous.com` |

Copy environment variables from [`nomicous/backend/core/.env.production.example`](../../nomicous/backend/core/.env.production.example) into the Vercel project (Production).

Set `JOB_WORKER_ENABLED=false` and `JOB_SSE_NOTIFICATIONS_ENABLED=false` on the
Vercel API. Run [`backend.jobs.worker_main`](../../nomicous/backend/jobs/worker_main.py)
on a persistent host instead (see [`deploy/inference/README.md`](../inference/README.md)).
`FORWARDED_ALLOW_IPS` must be an explicit proxy IP/CIDR allowlist; never use
`*`, and leave `BEHIND_PROXY=false` when the provider has no stable source range.

## Health check

After deploy:

```bash
curl -s https://api.nomicous.com/health
```
