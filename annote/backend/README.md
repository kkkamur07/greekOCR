# Annote Backend

FastAPI backend for the production Annote platform. It exposes authentication,
Project, Document, Document part/Page, layout, transcription, annotation history,
Export, Transcription PDF, ML model catalog, and job APIs.

Run backend commands from the `annote/` app root unless a command says otherwise.

## Quick Start

```bash
cp annote/backend/core/.env.example annote/backend/core/.env
docker compose up db -d
cd annote
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
  jobs/                           # enqueue, polling, workers, job persistence
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
| `jobs` | Async job enqueueing, status polling, claiming, worker execution, failure persistence | `jobs/api/jobs.py`, `jobs/application/job_service.py`, `jobs/infrastructure/worker.py` |

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
- `GET /inference/models`, model binding routes, `GET /jobs/{job_id}`
- Public read-only routes under `/public/...`

After API changes, regenerate frontend contracts:

```bash
cd annote
PYTHONPATH=. python scripts/export_openapi.py
cd frontend
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
| `MEDIA_ROOT` | Uploaded Document part media | `annote/backend/media` |
| `ENABLE_TEST_JOB_ROUTES` | Enables dev-only noop job route | `false` in `.env.example` |
| `TRANSCRIBE_ADAPTER` | Adapter marker used for transcribe jobs | `mock:transcribe` |
| `SEGMENT_ADAPTER` | Legacy segment job payload marker; execution uses `ML_SERVICE_URL` | `kraken_stub` |
| `ML_SERVICE_URL` | Root ML inference service used for segment jobs | `http://localhost:8001` |
| `DEFAULT_SEGMENT_MODEL` | Dev segment model name | `kraken-segment-default` |
| `DEFAULT_TRANSCRIBE_MODEL` | Dev transcribe model name | `kraken-transcribe-default` |
| `KRAKEN_MODEL_PATH` | Path to model weights outside annote | `../model/kraken` |

Keep real secrets out of git. `backend/core/.env` is local-only.

## Persistence and Media

Structured platform data is in Postgres. Uploaded page images are stored on disk
under `MEDIA_ROOT` and referenced by `document_parts.image_key`.

The repository-level `model/` workspace is intentionally outside `annote/`.
Inference model rows store artifact references; they do not copy weights into
the production app.

## Tests

Focused platform test examples:

```bash
cd annote
PYTHONPATH=. pytest backend/tests/platform/test_auth.py -q
PYTHONPATH=. pytest backend/tests/platform/test_documents.py -q
PYTHONPATH=. pytest backend/tests/platform/test_pairing_progress.py -q
PYTHONPATH=. pytest backend/tests/platform/test_annotation_history.py -q
PYTHONPATH=. pytest backend/tests/platform/test_export_approved_line_artifacts.py -q
PYTHONPATH=. pytest backend/tests/platform/test_transcription_pdf_artifact.py -q
```

Run the full platform suite:

```bash
cd annote
PYTHONPATH=. pytest backend/tests/platform
```

All backend API tests live under `backend/tests/platform/` and exercise the
platform FastAPI app.

## ML inference service

Segment jobs call the repository-level **`ml/` inference service** through
`ML_SERVICE_URL`. The backend owns product job enqueueing, polling, media
lookup, and canonical layout merge; the ML service owns model registry lookup
and Kraken segmentation execution.

Compose sets `ML_SERVICE_URL=http://ml-api:8001` on the API container. In
platform tests, `httpx.MockTransport` provides an in-process ML HTTP boundary so
the backend job and merge flow stays deterministic without bundling heavyweight
Kraken weights into the platform test suite.

Within the standalone service, `ml-api` is the HTTP boundary for health checks,
sync runs, and async job submission. `ml-worker` is the background executor for
queued model work, so segmentation/transcription can run without blocking API
request workers and can later scale onto different CPU/GPU resources.

## Special Notes

- The production platform is `backend/core` plus bounded contexts.
- Job workers are started by the FastAPI lifespan in `backend/core/app.py`.
- `Page transcription` candidate Text lines are helpers; Ground truth lives in
  `line_transcriptions` under a `ground_truth` Transcription layer.
- `Review status` is a boolean on `document_parts`, independent from Pairing
  progress.
- `Annotation history` stores compact JSON snapshots, not raw edit events or
  image/export bytes.
