# Nomicous Backend

FastAPI backend for the production Nomicous platform. It exposes authentication,
Project, Document, Document part/Page, layout, transcription, annotation history,
Export, Transcription PDF, ML model catalog, and job APIs.

Run backend commands from the `nomicous/` app root unless a command says otherwise.

## Quick Start

```bash
cp nomicous/backend/core/.env.example nomicous/backend/core/.env
docker compose up db -d
cd nomicous
uv run --project .. --group platform alembic -c infrastructure/alembic.ini upgrade head
uv run --project .. --group platform uvicorn backend.core.app:create_app --factory --reload
```

API URLs:

- App: `http://localhost:8000`
- Health: `http://localhost:8000/health`
- OpenAPI UI: `http://localhost:8000/docs`

## Directory Map

```text
backend/
  core/                           # FastAPI composition, settings, shared errors
    app.py                        # create_app(), middleware, router wiring
    main.py                       # Uvicorn module entrypoint
    api/                          # health/root routes
    settings/                     # Pydantic settings split by concern
  users/                          # register/login/me, JWT, password hashing
  project/                        # project CRUD, sharing, membership checks
  document/                       # documents, parts, layout, transcriptions
  annotation/                     # history snapshots, export, PDF artifacts
  ml/                             # model catalog, bindings, ML adapters
  jobs/                           # enqueue, status, NOTIFY→SSE push, workers, job persistence
  tests/platform/                 # Postgres-backed platform integration tests
```

## Architecture

The production backend uses a bounded-context layout:

```text
backend/<context>/
  api/                 # FastAPI routers and HTTP DTOs
  application/         # use cases and transaction orchestration
  domain/              # domain rules that do not need infrastructure
  infrastructure/      # SQLAlchemy ORM models and repositories/adapters
```

Important rules:

- `backend/core/app.py` wires routers; route behavior lives in context routers.
- Application services enforce access checks before touching context data.
- ORM metadata is aggregated in `infrastructure/models.py` for Alembic.
- FastAPI DTOs live beside routers in `api/schemas.py`.
- Tests should prefer public API behavior through `TestClient`.

## Contexts

| Context | Main responsibilities | Key files |
|---------|-----------------------|-----------|
| `users` | Registration, login, JWT auth, current-user dependency | `users/api/auth.py`, `users/application/auth_service.py` |
| `project` | Project CRUD, owner/shared-user membership | `project/api/projects.py`, `project/domain/access.py` |
| `document` | Documents, Document parts, media, layout Blocks/Lines, Transcriptions, Pairing progress | `document/api/documents.py`, `document/application/document_service.py` |
| `annotation` | Annotation history, Export artifacts, Transcription PDF artifacts | `annotation/api/history.py`, `annotation/application/export_service.py`, `annotation/application/transcription_pdf_service.py` |
| `ml` | ML model catalog, model bindings, ML service client, canonical model outputs | `ml/api/models.py`, `ml/application/model_service.py`, `ml/infrastructure/ml_client.py` |
| `jobs` | Async job enqueueing, status reads (`GET /jobs/{id}`), SSE push (`GET /jobs/{id}/events`), claiming, worker execution, failure persistence | `jobs/api/jobs.py`, `jobs/application/job_service.py`, `jobs/infrastructure/notifications.py`, `jobs/infrastructure/worker.py` |

## Job status notifications

Platform jobs (segment, transcribe, test noop) push status updates to the browser
with **Postgres `NOTIFY` for detection** and **Server-Sent Events (SSE) for
delivery**. Polling (`GET /jobs/{id}`) remains as a fallback when SSE is
unavailable.

Design background: [`docs/decisions/001-platform-job-status-push.md`](../../docs/decisions/001-platform-job-status-push.md).

### End-to-end flow

```mermaid
flowchart TB
  subgraph writers [Status writers]
    W[Platform job worker]
    CB[Inference callback]
    JR[job_repository helpers]
  end

  subgraph postgres [Postgres]
    JT[(jobs table)]
    CH[[NOTIFY platform_jobs]]
  end

  subgraph api [nomicous-api process]
    EMIT[notify_platform_job_status_changed]
    LOOP[platform_job_notification_loop<br/>asyncpg LISTEN]
    BROAD[JobStatusBroadcaster]
    SSE[GET /jobs/id/events]
  end

  subgraph browser [Browser]
    FE[watchJobViaSse / useJobPolling]
    POLL[GET /jobs/id poll fallback]
  end

  W --> JR
  CB --> JR
  JR -->|UPDATE + commit| JT
  JR --> EMIT
  EMIT --> CH
  CH --> LOOP
  LOOP --> BROAD
  BROAD --> SSE
  SSE --> FE
  POLL -.->|if SSE unavailable| FE
```

**Segment / transcribe job (happy path):**

```mermaid
sequenceDiagram
  participant B as Browser
  participant API as nomicous-api
  participant PG as Postgres
  participant INF as inference-worker

  B->>API: POST …/segment or …/transcribe
  API-->>B: job_id
  API->>PG: claim job → running
  Note over API,PG: pg_notify(platform_jobs)
  B->>API: GET /jobs/{id}/events (SSE)
  API-->>B: event: job (running)
  API->>INF: POST /inference/v1/jobs
  API->>PG: status → waiting
  Note over API,PG: pg_notify(platform_jobs)
  API-->>B: event: job (waiting)
  INF->>INF: Kraken / Calamari
  INF->>API: POST /internal/inference/job-complete
  API->>PG: merge result → done
  Note over API,PG: pg_notify(platform_jobs)
  API-->>B: event: job (done)
```

**Detection vs delivery** (two separate layers):

```mermaid
flowchart LR
  subgraph detect [1. Detection]
    A[Job row committed] --> B[pg_notify]
    B --> C[API LISTEN loop]
  end
  subgraph deliver [2. Delivery]
    D[In-process broadcaster] --> E[SSE stream]
    E --> F[Browser UI]
  end
  C --> D
```

1. **Emit** — After a committed `jobs` status change, code calls
   `notify_platform_job_status_changed()` (`jobs/infrastructure/notifications.py`),
   which runs `SELECT pg_notify(:channel, :payload)` on a sync session.
2. **Listen** — The FastAPI lifespan starts `platform_job_notification_loop()` in
   `backend/core/app.py`. It opens a dedicated **asyncpg** connection (using
   `SYNC_DATABASE_URL`, not the SQLAlchemy `postgresql+asyncpg://` URL) and
   `LISTEN`s on `PLATFORM_JOB_NOTIFY_CHANNEL` (default `platform_jobs`).
3. **Fan-out** — Each notification is parsed and passed to
   `job_status_broadcaster`, an in-memory registry of per-job `asyncio.Queue`s
   owned by active SSE handlers in that API process.
4. **Deliver** — `GET /jobs/{job_id}/events` (`jobs/api/jobs.py`) subscribes the
   client queue, sends the current `JobResponse` snapshot immediately, then
   streams further `job` events when the broadcaster receives NOTIFY payloads.
   Heartbeat comments (`: heartbeat`) are sent every `JOB_SSE_HEARTBEAT_SECONDS`
   (default 30) while waiting. The stream closes after `done` or `failed`.

### When NOTIFY fires

| Trigger | Location | Typical new status |
|---------|----------|-------------------|
| Platform worker claims a job | `jobs/infrastructure/job_repository.py` → `claim_next_pending_job` | `running` |
| Job submitted to inference | `mark_job_waiting` | `waiting` |
| Job completes | `mark_job_done` | `done` |
| Job fails | `mark_job_failed` | `failed` |
| Inference callback updates product job | `jobs/application/job_callback_service.py` | `done` / `failed` |

Test noop jobs follow the same path when the in-process worker handles them.

### Frontend consumption

The React app opens `GET /jobs/{job_id}/events` via `watchJobViaSse` /
`waitForJob` (`nomicous/frontend/src/utils/jobPolling.ts`). If SSE cannot be
opened, it falls back to polling `GET /jobs/{id}` every 250 ms. Background job
panels use `useJobPolling`, which prefers SSE and degrades to 1.5 s polling per
job when needed.

### Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `PLATFORM_JOB_NOTIFY_CHANNEL` | Postgres `NOTIFY` channel name | `platform_jobs` |
| `JOB_SSE_HEARTBEAT_SECONDS` | SSE keep-alive interval while idle | `30` |
| `SYNC_DATABASE_URL` | DSN for the asyncpg notification listener | `postgresql://…` |

The notification listener must use a plain `postgresql://` DSN (`SYNC_DATABASE_URL`).
`DATABASE_URL` (`postgresql+asyncpg://…`) is for SQLAlchemy only.

### Operations

- On startup, a healthy listener logs:
  `Listening for platform job notifications on platform_jobs`.
- If the listener fails to connect, the API still serves requests; clients rely on
  poll fallback until the loop reconnects (1 s backoff between attempts).
- Each API process has its own listener and broadcaster. `NOTIFY` is
  database-global, so every replica wakes; only SSE clients connected to that
  process receive the fan-out. Sticky sessions or a single worker avoid split
  clients in multi-replica deployments.

### Tests

- `tests/nomicous/integration/test_jobs.py` — `GET /jobs/{id}/events` auth and
  snapshot streaming.

**Inference worker notifications (separate channel):** The `inference/` service uses
its own `inference_jobs` Postgres channel to wake `inference-worker`. That path
does not talk to the browser; the platform callback updates `jobs` and triggers
`platform_jobs` NOTIFY above.

## API Surface

Major route families:

- `POST /auth/register`, `POST /auth/login`, `GET /auth/me`
- `GET/POST /projects`, project update/share routes
- `GET/POST /projects/{project_id}/documents`
- `POST /projects/{project_id}/documents/{document_id}/parts`
- `GET/PUT /projects/{project_id}/documents/{document_id}/parts/{part_id}/lines`
- `PUT /.../page-transcription`, `GET /.../pairing`, `POST /.../pairings`
- `PATCH /.../transcriptions/{transcription_id}/lines/{line_id}`
- `POST/GET /.../parts/{part_id}/history`, `POST /.../history/{snapshot_id}/restore`
- `POST /.../parts/{part_id}/export`
- `POST /.../parts/{part_id}/transcription-pdf`
- `GET /inference/models`, model binding routes, `GET /jobs/{job_id}`, `GET /jobs/{job_id}/events`
- Public read-only routes under `/public/...`

After API changes, regenerate frontend contracts:

```bash
# from repository root
python scripts/platform/export_openapi.py
cd nomicous/frontend
npm run codegen:api
```

## Settings

Settings are loaded by `backend/core/settings/` from environment variables and
`backend/core/.env`.

| Variable | Purpose | Local default |
|----------|---------|---------------|
| `DATABASE_URL` | Async SQLAlchemy app connection | `postgresql+asyncpg://postgres:dev@localhost:5433/kalamos` |
| `SYNC_DATABASE_URL` | Alembic sync connection | `postgresql://postgres:dev@localhost:5433/kalamos` |
| `JWT_SECRET` | JWT signing secret | required; use a unique value per environment |
| `JWT_EXPIRE_MINUTES` | Access-token lifetime | `60` |
| `AUTH_RATE_LIMIT_REQUESTS` | Login/register attempts per window per client | `60` |
| `AUTH_RATE_LIMIT_WINDOW_SECONDS` | Login/register rate-limit window | `60` |
| `CORS_ORIGINS` | Browser origins | `http://localhost:3000,http://localhost:5173` |
| `MEDIA_ROOT` | Uploaded Document part media | `nomicous/backend/media` |
| `ENABLE_TEST_JOB_ROUTES` | Enables dev-only noop job route | `false` in `.env.example` |
| `PLATFORM_JOB_NOTIFY_CHANNEL` | Postgres channel for platform job status NOTIFY | `platform_jobs` |
| `JOB_SSE_HEARTBEAT_SECONDS` | SSE idle heartbeat interval for `/jobs/{id}/events` | `30` |
| `JOB_WORKER_ENABLED` | Start in-process platform job worker on API boot | `true` |
| `TRANSCRIBE_ADAPTER` | Adapter marker used for transcribe jobs | `calamari` |
| `SEGMENT_ADAPTER` | Adapter marker used for segment jobs | `kraken` |
| `DEFAULT_SEGMENT_MODEL` | Dev segment model name | `kraken-segment-default` |
| `DEFAULT_TRANSCRIBE_MODEL` | Dev transcribe model name | `kraken-transcribe-default` |
| `KRAKEN_MODEL_PATH` | Path to model weights outside nomicous | `../model/kraken` |

Keep real secrets out of git. `backend/core/.env` is local-only.

## Persistence and Media

Structured platform data is in Postgres. Uploaded page images are stored on disk
under `MEDIA_ROOT` and referenced by `document_parts.image_key`.

The repository-level `model/` workspace is intentionally outside `nomicous/`.
Inference model rows store artifact references; they do not copy weights into
the production app.

## Tests

Focused platform test examples:

```bash
cd nomicous
PYTHONPATH=. pytest backend/tests/platform/test_auth.py -q
PYTHONPATH=. pytest backend/tests/platform/test_documents.py -q
PYTHONPATH=. pytest backend/tests/platform/test_pairing_progress.py -q
PYTHONPATH=. pytest backend/tests/platform/test_annotation_history.py -q
PYTHONPATH=. pytest backend/tests/platform/test_export_approved_line_artifacts.py -q
PYTHONPATH=. pytest backend/tests/platform/test_transcription_pdf_artifact.py -q
```

Run the full platform suite:

```bash
cd nomicous
PYTHONPATH=. pytest backend/tests/platform
```

All backend API tests live under `backend/tests/platform/` and exercise the
platform FastAPI app.

## ML inference service

Segment and transcribe jobs call the repository-level **`inference/` service** through `InferenceClient` (`backend/ml/infrastructure/ml_client.py`).

Compose sets `INFERENCE_URL=http://inference-api:8001` on the API container. The standalone service (health, sync `/inference/v1/run`, async job submission, contracts, registry — see [`inference/README.md`](../../inference/README.md)) runs as `inference-api` + `inference-worker`.

## Special Notes

- The production platform is `backend/core` plus bounded contexts.
- Job workers and the Postgres notification listener are started by the FastAPI
  lifespan in `backend/core/app.py` (see **Job status notifications** above).
- `Page transcription` candidate Text lines are helpers; Ground truth lives in
  `line_transcriptions` under a `ground_truth` Transcription layer.
- `Review status` is a boolean on `document_parts`, independent from Pairing
  progress.
- `Annotation history` stores compact JSON snapshots, not raw edit events or
  image/export bytes.
