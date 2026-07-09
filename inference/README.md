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

## Weights layout

### Local bundled weights (`src/hf/local/`)

Bundled checkpoints for offline dev and Docker live under `src/hf/local/{script}/{architecture}/{model_version}/{registry_tag}/`. Registry entries reference them with `file://` URIs relative to `src/hf/` (for example `file://local/syriac/calamari/v1/stable/best.pt`).

Docker Compose mounts `./src/hf` at `/app/src/hf` on `inference-api` and `inference-worker`. No Hub credentials are required for models that use local bundled weights.

**Migrated:** `syriac-calamari-v1` (`stable`) — served from `src/hf/local/syriac/calamari/v1/stable/`.

### Interim layout (`inference/weights/`)

Legacy checkpoints still under `inference/weights/` use `file://` URIs relative to `inference/` (interim layout). Docker Compose continues to mount `inference/weights/` at `/app/inference/weights`. These paths will move to `src/hf/local/` as models are migrated.

Runtime weight cache for resolved remote weights: `src/hf/cache/<registry_model_id>/<registry_tag>/` (Hub integration). Interim `inference/weights/cache/` remains the default for `INFERENCE_WEIGHTS_CACHE_DIR` until all models use Hub cache.

### Calamari (PyTorch runtime)

Transcribe uses the local PyTorch Calamari implementation under `inference/architectures/calamari/`.
Runtime artifacts are converted `.pt` checkpoints, so inference images do not install TensorFlow
or copy the vendored Calamari source tree.

Historical migration notes: [`docs/architecture/calamari-vendored-architecture.md`](../docs/architecture/calamari-vendored-architecture.md).

**Hub integration:** `hf://` weight sources, Hub cache, and prefetch tooling live under `src/hf/` and `scripts/hf/`. See `inference/CONTEXT.md` for domain terminology and [`scripts/hf/README.md`](../scripts/hf/README.md) for the Hub publish runbook.

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
| `INFERENCE_WEIGHTS_CACHE_DIR` | `inference/weights/cache` | Interim runtime cache directory (Hub cache uses `src/hf/cache/`) |
| `HF_TOKEN` | unset | Required only for **private** or gated Hub repos; all nomicous inference repos are public |

Prefetch Hub weights without running inference:

```bash
PYTHONPATH=. python scripts/hf/fetch_model.py greek-calamari-v1 --registry-tag stable
```

## Contracts

Shared Pydantic schemas in `inference/contracts/` define the wire format for inference endpoints:

- **Run** — `InferenceRunRequest` / `InferenceRunResponse` (`inference/contracts/run.py`): task, registry model, image bytes, and params in; typed output out.
- **Segment** — `SegmentRunResponse` (`inference/contracts/segment.py`): page image in, blocks and line polygons out.
- **Transcribe** — `TranscribeRunResponse` / `TranscribeBatchRunResponse` (`inference/contracts/transcribe.py`): line image(s) in, text and per-character confidence out.

Both tasks reference models by `registry_model_id` and optional `registry_tag` (default `stable`).

Job callbacks use a tagged output union: `output.kind` is either `segment` or `transcribe`, and `output.data` contains the matching result schema. Invalid callback shapes, such as a `done` callback with an `error` field, missing output, or a `task`/`output.kind` mismatch, are request-body validation failures. When an endpoint accepts `JobCallbackRequest` directly, FastAPI should return **422 Unprocessable Entity** for those cases. Use **404 Not Found** only for runtime lookups such as an unknown job id or unknown `registry_model_id`.

## Registry

`inference/registry.yaml` lists available models and weight locations. Example entries:

- `greek-calamari-v1` — transcribe, Calamari architecture
- `greek-kraken-segment-v1` — segment, Kraken BLLA

Weights are resolved at runtime from `src/hf/local/` (bundled), `src/hf/cache/` (Hub), `inference/weights/` (interim), or `package://` (Kraken).

**Adding a model:** step-by-step checklist in [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

## Inference helper (local CPU on researcher machines)

For hosted SPA + local inference, run the slim helper sidecar (no Postgres, no job queue):

```bash
HELPER_REGISTRY_URL=http://localhost:8000/inference/v1/registry \
HF_CACHE_ROOT=~/.nomicous/hf/cache uv run --group inference python -m inference.helper
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/inference/v1/catalog
```

On startup the helper fetches `registry.yaml` from the hosted platform (`GET /inference/v1/registry`, public, ETag-aware) into `~/.nomicous/registry.yaml`. The bundled copy in the installer is only a fallback when offline. Model **weights** still download on first `/run` via existing `hf://` resolution.

The browser probes `localhost:8001`, calls `/inference/v1/run`, and persists results through the hosted platform API. Set `HELPER_CORS_ORIGINS` to your SPA origin(s).

Packaging for `.dmg` / `.msi` / Linux installers: [`packaging/helper/README.md`](../packaging/helper/README.md) — PyInstaller spec excludes training stacks, platform API, and unused torch backends so installers ship only Calamari + Kraken runtime.

## Tests

```bash
uv run --group inference pytest tests/inference tests/hf
```

Stop the Compose `inference-worker` before local integration runs (`docker stop nomicous-inference-worker-1`).
Full-suite layout, `DATABASE_URL` caveats, and failure analysis: [`docs/guides/testing.md`](../docs/guides/testing.md).

## Related docs

- Nomicous platform API and job integration: [`nomicous/backend/README.md`](../nomicous/backend/README.md)
- Compose stack and env vars: [`docker-compose.yml`](../docker-compose.yml) and [`nomicous/README.md`](../nomicous/README.md)
