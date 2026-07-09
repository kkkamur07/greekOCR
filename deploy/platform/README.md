# Platform API — Vercel deploy

Serverless FastAPI bundle for **api.nomicous.com**. Full production runbook: [`docs/deployment/production.md`](../../docs/deployment/production.md).

## What this folder is

Vercel's **Root Directory** for the `nomicous-api` project. At build time `build.sh` copies the platform backend and inference registry metadata into this tree; generated artifacts are gitignored.

| Path | Purpose |
|------|---------|
| `api/index.py` | ASGI entrypoint (`app` export) |
| `build.sh` | Bundle sources + export `requirements.txt` |
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
| Install Command | `pip install -r requirements.txt` |
| Build Command | `bash build.sh` |
| Domain | `api.nomicous.com` |

Copy environment variables from [`nomicous/backend/core/.env.production.example`](../../nomicous/backend/core/.env.production.example) into the Vercel project (Production).

**Do not** set `JOB_WORKER_ENABLED=true` on Vercel — run [`backend.jobs.worker_main`](../../nomicous/backend/jobs/worker_main.py) on a persistent host instead (see [`deploy/inference/README.md`](../inference/README.md)).

## Health check

After deploy:

```bash
curl -s https://api.nomicous.com/health
```
