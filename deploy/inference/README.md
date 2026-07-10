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
INFERENCE_CALLBACK_URL=https://api.nomicous.com/internal/inference/job-complete
INFERENCE_REGISTRY_PATH=/app/inference/registry.yaml
INFERENCE_WEIGHTS_CACHE_DIR=/app/inference/weights/cache
```

Configure connection URLs only in the host's secret manager:

| Service | Required DB environment | Database group role |
|---|---|---|
| Inference API + worker | `INFERENCE_DATABASE_URL` | `nomicous_inference_worker` |
| Platform worker | `DATABASE_URL`, `SYNC_DATABASE_URL` | `nomicous_platform_worker` |
| Platform API | `DATABASE_URL`, `SYNC_DATABASE_URL` | `nomicous_api` |
| Migration job | `MIGRATOR_DATABASE_URL` | `nomicous_migrator` / provider operator |

Set `INFERENCE_WEBHOOK_SECRET` and `INFERENCE_SERVICE_SECRET` from the same
secret store; they must match the platform API's values. The inference services
must not receive the API, storage, JWT, or migration database URL.
In production, both secrets and `INFERENCE_CALLBACK_URL` are required at
startup; placeholder values are rejected. Callback URLs must use HTTPS except
for `localhost`, `127.0.0.1`, and the Docker Compose `api` / `inference-api`
service hosts.

Platform worker additionally needs its platform DB URLs and inference endpoint:

```bash
INFERENCE_URL=https://inference.nomicous.com
```

Bootstrap roles and provider logins before deployment with
[`docs/deployment/database-roles.md`](../../docs/deployment/database-roles.md).
Use `nomicous/backend/core/.env.platform-worker.example` as the persistent
platform-worker secret template and `nomicous/backend/core/.env.inference.example`
for inference API/worker secrets.

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

---

## Helper release evidence

The `Release Inference Helper` workflow builds installers per OS only for an
existing `inference-helper-v*` GitHub Release. It verifies each installer
checksum before publishing and deliberately refuses to replace an existing
release asset. Use a new tag for a corrected release; never use `--clobber`.

Each release includes `SHA256SUMS` and an SPDX SBOM. Public repositories also
receive GitHub build provenance attestations. Verify a downloaded installer
before distribution:

```bash
shasum -a 256 -c SHA256SUMS
gh attestation verify <installer-file> --repo <owner>/<repository>
```

For private repositories, GitHub artifact attestations require a compatible
GitHub plan; the workflow skips the attestation rather than weakening its
permissions or using a long-lived signing secret.
