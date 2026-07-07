# ML inference service

Standalone FastAPI service for manuscript **segment** and **transcribe** inference. It lives at the repository root in `inference/`, separate from the Nomicous platform API in `nomicous/backend/`.

## Status

| Piece | State |
|-------|--------|
| HTTP API (`inference/api/`) | Health, sync `/inference/v1/run`, and async `/inference/v1/jobs` submission |
| Request/response contracts (`inference/contracts/`) | Defined for segment, transcribe, jobs, and callbacks |
| Model registry (`inference/registry.yaml`) | Calamari transcribe + Kraken segment entries |
| Worker (`inference/jobs/worker.py`) | Postgres-backed queue worker with LISTEN/NOTIFY wakeups |
| Nomicous backend integration | Platform jobs delegate segment/transcribe work via `InferenceClient` |

The root `docker-compose.yml` starts `inference-api` and `inference-worker` alongside the platform API.

## Docker Compose

| Service | Port | Role |
|---------|------|------|
| `inference-api` | 8001 | Inference HTTP API |
| `inference-worker` | — | Background job processor |

## API vs worker

`inference-api` is the HTTP-facing boundary. It owns health checks, synchronous `/inference/v1/run`, and async job submission (`POST /inference/v1/jobs`). It stays responsive even when model work is slow.

`inference-worker` is the background executor for long-running CPU/GPU work: Kraken segmentation, Calamari transcription, model loading, retries, and posting job callbacks.

Keeping them separate lets the API and workers scale, restart, and fail independently. Workers can later run on different resources (e.g. GPU nodes) without changing the HTTP contract.

```bash
docker compose up --build
curl -s http://localhost:8001/health
```

## Weights layout (interim)

Today, bundled checkpoints live under `inference/weights/` and are referenced from `registry.yaml` via `file://` or `package://` URIs. Docker Compose mounts `inference/weights/` at `/app/inference/weights`. Registry path defaults to `/app/inference/registry.yaml`.

### Calamari (vendored, not PyPI)

Transcribe uses a **vendored** Calamari tree (`src/model/calamari`), copied into the image at `/app/_support_repo/calamari`. We do not import the PyPI `calamari-ocr` package as the runtime `calamari_ocr` source. TensorFlow and related deps still come from the `inference` uv group.

Full migration notes, Docker rebuild steps, and troubleshooting: [`docs/calamari-vendored-architecture.md`](../docs/calamari-vendored-architecture.md).

**Target architecture (planned, not implemented):** Hub integration under `src/hf/`, remote `hf://` weight sources, Hub cache under `src/hf/cache/`, publish tooling under `scripts/hf/`, and local bundled weights under `src/hf/local/`. See `inference/CONTEXT.md` for domain terminology.

## Run locally (without Compose)

From the repository root, with the `inference` dependency group installed:

```bash
uv sync --group inference
PYTHONPATH=. uvicorn inference.api.main:app --host 0.0.0.0 --port 8001 --reload
```

Environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERENCE_REGISTRY_PATH` | `inference/registry.yaml` | Model catalog file |
| `INFERENCE_WEIGHTS_CACHE_DIR` | `inference/weights/cache` | Runtime cache directory for resolved weights |

## Contracts

Shared Pydantic schemas in `inference/contracts/` define the wire format for inference endpoints:

- **Run** — `InferenceRunRequest` / `InferenceRunResponse` (`inference/contracts/run.py`): task, registry model, image bytes, and params in; typed output out.
- **Segment** — `SegmentRunResponse` (`inference/contracts/segment.py`): page image in, blocks and line polygons out.
- **Transcribe** — `TranscribeRunResponse` / `TranscribeBatchRunResponse` (`inference/contracts/transcribe.py`): line image(s) in, text and per-character confidence out.

Both tasks reference models by `registry_model_id` and optional `registry_tag` (default `stable`).

Job callbacks use a tagged output union: `output.kind` is either `segment` or `transcribe`, and `output.data` contains the matching result schema. Invalid callback shapes, such as a `done` callback with an `error` field, missing output, or a `task`/`output.kind` mismatch, are request-body validation failures. When an endpoint accepts `JobCallbackRequest` directly, FastAPI should return **422 Unprocessable Entity** for those cases. Use **404 Not Found** only for runtime lookups such as an unknown job id or unknown `registry_model_id`.

## Registry

`inference/registry.yaml` lists available models and weight locations. Example entries:

- `greek-calamariv1` — transcribe, Calamari architecture
- `kraken-blla` — segment, Kraken BLLA

Weights are resolved at runtime from `inference/weights/` (interim layout; not copied into Nomicous's Postgres catalog).

## Tests

```bash
uv run --group inference pytest tests/inference
```

Stop the Compose `inference-worker` before local integration runs (`docker stop nomicous-inference-worker-1`).
Full-suite layout, `DATABASE_URL` caveats, and failure analysis: [`docs/testing.md`](../docs/testing.md).

## Related docs

- Nomicous platform API and job integration: [`nomicous/backend/README.md`](../nomicous/backend/README.md)
- Compose stack and env vars: [`docker-compose.yml`](../docker-compose.yml) and [`nomicous/README.md`](../nomicous/README.md)
