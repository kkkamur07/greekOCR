# Inference production deploy

Cloud ML inference **does not run on Vercel**. Use the existing Docker image on a host with enough CPU/RAM and persistent disk for model weights.

Recommended: **Railway** or **Fly.io** with three processes from the same image.

---

## Services

### 1. Inference API

```bash
uvicorn inference.api.main:app --host 0.0.0.0 --port 8001 --workers 1
```

Public URL: `https://inference.nomicous.com`

### 2. Inference worker

```bash
python -m inference.jobs.worker
```

No public port. Polls `inference_jobs` via Postgres `NOTIFY`.

### 3. Platform job worker

Dispatches platform `segment` / `transcribe` jobs to the inference API. Required because the Vercel API sets `JOB_WORKER_ENABLED=false`.

```bash
cd nomicous && python -m backend.jobs.worker_main
```

Or from repo root:

```bash
PYTHONPATH=nomicous:. python -m backend.jobs.worker_main
```

---

## Environment variables

Shared across all three services (inference + platform worker):

```bash
INFERENCE_DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres?sslmode=require
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres?sslmode=require

INFERENCE_CALLBACK_URL=https://api.nomicous.com/internal/inference/job-complete
INFERENCE_WEBHOOK_SECRET=<same as platform API>
INFERENCE_SERVICE_SECRET=<same as platform API>
INFERENCE_REGISTRY_PATH=/app/inference/registry.yaml
INFERENCE_WEIGHTS_CACHE_DIR=/app/inference/weights/cache
```

Platform worker additionally needs platform DB URLs (same Supabase project):

```bash
DATABASE_URL=postgresql+asyncpg://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres?sslmode=require
SYNC_DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres?sslmode=require
INFERENCE_URL=https://inference.nomicous.com
```

---

## Docker Compose (reference)

For a single VM or Railway with Docker:

```yaml
services:
  inference-api:
    image: nomicous-inference:latest
    build:
      context: ..
      dockerfile: inference/Dockerfile
    ports:
      - "8001:8001"
    environment:
      INFERENCE_DATABASE_URL: ${INFERENCE_DATABASE_URL}
      INFERENCE_CALLBACK_URL: https://api.nomicous.com/internal/inference/job-complete
      INFERENCE_WEBHOOK_SECRET: ${INFERENCE_WEBHOOK_SECRET}
      INFERENCE_SERVICE_SECRET: ${INFERENCE_SERVICE_SECRET}
    command: uvicorn inference.api.main:app --host 0.0.0.0 --port 8001

  inference-worker:
    image: nomicous-inference:latest
    environment:
      INFERENCE_DATABASE_URL: ${INFERENCE_DATABASE_URL}
      INFERENCE_CALLBACK_URL: https://api.nomicous.com/internal/inference/job-complete
      INFERENCE_WEBHOOK_SECRET: ${INFERENCE_WEBHOOK_SECRET}
      INFERENCE_SERVICE_SECRET: ${INFERENCE_SERVICE_SECRET}
    command: python -m inference.jobs.worker

  platform-worker:
    image: nomicous-api:latest
    build:
      context: ..
      dockerfile: nomicous/Dockerfile
    environment:
      DATABASE_URL: ${DATABASE_URL}
      SYNC_DATABASE_URL: ${SYNC_DATABASE_URL}
      INFERENCE_URL: https://inference.nomicous.com
      INFERENCE_WEBHOOK_SECRET: ${INFERENCE_WEBHOOK_SECRET}
      INFERENCE_SERVICE_SECRET: ${INFERENCE_SERVICE_SECRET}
      JOB_WORKER_ENABLED: "true"
    command: python -m backend.jobs.worker_main
```

---

## Railway quick start

1. Create a project from this repo.
2. Add service **inference-api** — Dockerfile path `inference/Dockerfile`, start command as above.
3. Add service **inference-worker** — same image, worker start command.
4. Add service **platform-worker** — Dockerfile `nomicous/Dockerfile`, `python -m backend.jobs.worker_main`.
5. Attach a volume to inference services for `INFERENCE_WEIGHTS_CACHE_DIR` (models download on first run).
6. Map custom domain `inference.nomicous.com` to inference-api.

---

## Health checks

```bash
curl https://inference.nomicous.com/health
curl https://api.nomicous.com/health
```

Submit a transcribe job from the page editor on `app.nomicous.com` with cloud inference enabled to verify the full pipeline.
