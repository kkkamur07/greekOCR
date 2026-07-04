# ML inference service

Standalone FastAPI service for manuscript **segment** and **transcribe** inference. It lives at the repository root in `ml_service/`, separate from the Annote platform API in `annote/backend/`.

## Status

| Piece | State |
|-------|--------|
| HTTP API (`ml_service/api/`) | Health, sync run, and async job submission |
| Request/response contracts (`ml_service/contracts/`) | Defined for segment, transcribe, jobs, and callbacks |
| Model registry (`ml_service/registry.yaml`) | Calamari transcribe + Kraken segment entries |
| Worker (`ml_service/jobs/worker.py`) | Postgres-backed queue worker with LISTEN/NOTIFY wakeups |
| Annote backend integration | Jobs delegate segment/transcribe work to the ML service |

The root `docker-compose.yml` starts `ml-api` and `ml-worker` alongside the platform API.

## Docker Compose

| Service | Port | Role |
|---------|------|------|
| `ml-api` | 8001 | Inference HTTP API |
| `ml-worker` | — | Background job processor |

## API vs worker

`ml-api` is the HTTP-facing boundary. It owns health checks today and will own future inference endpoints, job submission, and status/result lookup. It should stay responsive even when model work is slow.

`ml-worker` is the background executor. It is where long-running CPU/GPU work belongs: Kraken segmentation, Calamari transcription, model loading, retries, and writing job results.

Keeping them separate lets the API and workers scale, restart, and fail independently. It also gives us a path to run workers on different resources later, such as GPU nodes, without changing the HTTP contract. The worker is a placeholder today, but the split is intentional for async ML jobs.

```bash
docker compose up --build
curl -s http://localhost:8001/health
```

Weights are mounted from `ml_service/weights/` at `/app/ml_service/weights`. Registry path defaults to `/app/ml_service/registry.yaml`.

## Run locally (without Compose)

From the repository root, with the `ml` dependency group installed:

```bash
uv sync --group ml
PYTHONPATH=. uvicorn ml_service.api.main:app --host 0.0.0.0 --port 8001 --reload
```

Environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ML_REGISTRY_PATH` | `ml_service/registry.yaml` | Model catalog file |
| `ML_WEIGHTS_CACHE_DIR` | `ml_service/weights/cache` | Download/cache directory for weights |

## Contracts

Shared Pydantic schemas in `ml_service/contracts/` define the wire format for inference endpoints:

- **Segment** — `SegmentRunRequest` / `SegmentRunResponse` (`ml_service/contracts/segment.py`): page image in, blocks and line polygons out.
- **Transcribe** — `TranscribeRunRequest` / `TranscribeRunResponse` (`ml_service/contracts/transcribe.py`): line image in, text and per-character confidence out.

Both tasks reference models by `registry_model_id` and optional `registry_tag` (default `stable`).

Job callbacks use a tagged output union: `output.kind` is either `segment` or `transcribe`, and `output.data` contains the matching result schema. Invalid callback shapes, such as a `done` callback with an `error` field, missing output, or a `task`/`output.kind` mismatch, are request-body validation failures. When an endpoint accepts `JobCallbackRequest` directly, FastAPI should return **422 Unprocessable Entity** for those cases. Use **404 Not Found** only for runtime lookups such as an unknown job id or unknown `registry_model_id`.

## Registry

`ml_service/registry.yaml` lists available models and weight locations. Example entries:

- `greek-calamariv1` — transcribe, Calamari architecture
- `kraken-blla` — segment, Kraken BLLA

Weights live under `ml_service/weights/` (not copied into Annote's Postgres catalog).

## Tests

```bash
uv run --group ml pytest ml_service/tests
```

## Related docs

- Annote platform API and job adapters: [`annote/backend/README.md`](../annote/backend/README.md)
- Compose stack and env vars: [`docker-compose.yml`](../docker-compose.yml) and [`annote/README.md`](../annote/README.md)
