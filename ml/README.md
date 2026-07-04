# ML inference service

Standalone FastAPI service for manuscript **segment** and **transcribe** inference. It lives at the repository root in `ml/`, separate from the Annote platform API in `annote/backend/`.

## Status

| Piece | State |
|-------|--------|
| HTTP API (`ml/api/`) | Health, sync segment run, async job submit |
| Request/response contracts (`ml/contracts/`) | Defined for segment and transcribe |
| Model registry (`ml/registry.yaml`) | Calamari transcribe + Kraken segment entries |
| Worker (`ml/jobs/worker.py`) | PostgreSQL queue listener + callback delivery |
| Annote backend integration | Segment jobs use `ML_SERVICE_URL` |

The service is under active development. Segment execution is wired first;
transcription execution remains follow-on work.

## Annote integration

The root `docker-compose.yml` starts `ml-api` and `ml-worker` alongside the platform API and sets:

```yaml
ML_SERVICE_URL: http://ml-api:8001
```

on the `api` service. Segment jobs are called by the Annote platform through
that URL. Transcription execution still lands in follow-on slices.

## Docker Compose

| Service | Port | Role |
|---------|------|------|
| `ml-api` | 8001 | Inference HTTP API |
| `ml-worker` | — | Background job processor |

## API vs worker

`ml-api` is the HTTP-facing boundary. It owns health checks today and will own future inference endpoints, job submission, and status/result lookup. It should stay responsive even when model work is slow.

`ml-worker` is the background executor. It is where long-running CPU/GPU work belongs: Kraken segmentation, Calamari transcription, model loading, retries, and writing job results.

Keeping them separate lets the API and workers scale, restart, and fail independently. It also gives us a path to run workers on different resources later, such as GPU nodes, without changing the HTTP contract.

```bash
docker compose up --build
curl -s http://localhost:8001/health
```

Weights are mounted from `ml/weights/` at `/app/ml/weights`. Registry path defaults to `/app/ml/registry.yaml`.

## Run locally (without Compose)

From the repository root, with the `ml` dependency group installed:

```bash
uv sync --group ml
PYTHONPATH=. uvicorn ml.api.main:app --host 0.0.0.0 --port 8001 --reload
```

Environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ML_REGISTRY_PATH` | `ml/registry.yaml` | Model catalog file |
| `ML_WEIGHTS_CACHE_DIR` | `ml/weights/cache` | Download/cache directory for weights |

## Contracts

Shared Pydantic schemas in `ml/contracts/` define the wire format for future inference endpoints:

- **Segment** — `SegmentRunRequest` / `SegmentRunResponse` (`ml/contracts/segment.py`): page image in, blocks and line polygons out.
- **Transcribe** — `TranscribeRunRequest` / `TranscribeRunResponse` (`ml/contracts/transcribe.py`): line image in, text and per-character confidence out.

Both tasks reference models by `registry_model_id` and optional `registry_tag` (default `stable`).

Job callbacks use a tagged output union: `output.kind` is either `segment` or `transcribe`, and `output.data` contains the matching result schema. Invalid callback shapes, such as a `done` callback with an `error` field, missing output, or a `task`/`output.kind` mismatch, are request-body validation failures. When an endpoint accepts `JobCallbackRequest` directly, FastAPI should return **422 Unprocessable Entity** for those cases. Use **404 Not Found** only for runtime lookups such as an unknown job id or unknown `registry_model_id`.

## Registry

`ml/registry.yaml` lists available models and weight locations. Example entries:

- `greek-calamariv1` — transcribe, Calamari architecture
- `kraken-blla` — segment, Kraken BLLA

Weights live under `ml/weights/` (not copied into Annote's Postgres catalog).

## Tests

```bash
uv run --group ml pytest ml/tests
```

## Related docs

- Annote platform API and job adapters: [`annote/backend/README.md`](../annote/backend/README.md)
- Compose stack and env vars: [`docker-compose.yml`](../docker-compose.yml) and [`annote/README.md`](../annote/README.md)
