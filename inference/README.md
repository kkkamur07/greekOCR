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
| `inference-worker` | - | Background job processor |

## API vs worker

`inference-api` is the HTTP-facing boundary. It owns health checks, synchronous `/inference/v1/run`, and async job submission (`POST /inference/v1/jobs`). It stays responsive even when model work is slow.

`inference-worker` is the background executor for long-running CPU/GPU work: Kraken segmentation, Calamari transcription, model loading, retries, and posting job callbacks.

Keeping them separate lets the API and workers scale, restart, and fail independently. Workers can later run on different resources (e.g. GPU nodes) without changing the HTTP contract.

```bash
docker compose up --build
curl -s http://127.0.0.1:8010/health
```

## Weights layout

Registry models resolve weights at runtime from:

| Source | Example | Cache / path |
|--------|---------|----------------|
| Hub | `hf://kkkamur07/syriac-htr-calamari@stable` | `src/hf/cache/<registry_model_id>/<registry_tag>/` |
| Local bundled (offline) | `file://local/syriac/calamari/v1/stable/best.pt` | `src/hf/local/...` |
| Kraken package | `package://kraken/blla.mlmodel` | Inside `kraken` pip package |

Docker Compose mounts `./src/hf` at `/app/src/hf` on `inference-api` and `inference-worker` and sets `HF_CACHE_ROOT=/app/src/hf/cache`. No local weight checkout is required for the default Hub models; they download from their public repos on first use. The default Kraken segment model is packaged with the `kraken` dependency.

### Calamari (PyTorch runtime)

Transcribe uses the local PyTorch Calamari implementation under `inference/architectures/calamari/`.
Runtime artifacts are converted `.pt` checkpoints (`calamari-pytorch-v1`), so inference images do not install TensorFlow
or copy the vendored Calamari source tree.

Training and vendored TensorFlow Calamari: [`docs/guides/learnings.md`](../docs/guides/learnings.md#calamari-training).

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
| `HF_CACHE_ROOT` | `src/hf/cache` | Hub weight download cache |
| `HF_TOKEN` | unset | Required only for **private** or gated Hub repos; all nomicous inference repos are public |

Prefetch Hub weights without running inference:

```bash
PYTHONPATH=. python scripts/hf/fetch_model.py syriac-calamari-v1 --registry-tag stable
```

## Contracts

Shared Pydantic schemas in `inference/contracts/` define the wire format for inference endpoints:

- **Run** - `InferenceRunRequest` / `InferenceRunResponse` (`inference/contracts/run.py`): task, registry model, image bytes, and params in; typed output out.
- **Segment** - `SegmentRunResponse` (`inference/contracts/segment.py`): page image in, blocks and line polygons out.
- **Transcribe** - `TranscribeRunResponse` / `TranscribeBatchRunResponse` (`inference/contracts/transcribe.py`): line image(s) in, text and per-character confidence out.

Both tasks reference models by `registry_model_id` and optional `registry_tag` (default `stable`).

Job callbacks use a tagged output union: `output.kind` is either `segment` or `transcribe`, and `output.data` contains the matching result schema. Invalid callback shapes, such as a `done` callback with an `error` field, missing output, or a `task`/`output.kind` mismatch, are request-body validation failures. When an endpoint accepts `JobCallbackRequest` directly, FastAPI should return **422 Unprocessable Entity** for those cases. Use **404 Not Found** only for runtime lookups such as an unknown job id or unknown `registry_model_id`.

## Registry

`inference/registry.yaml` lists available models and weight locations. Example entries:

- `syriac-calamari-v1` - transcribe, Calamari architecture, pinned Hub revision and digest
- `kraken-segment` - segment, Kraken BLLA package weights

Weights are resolved at runtime from Hub cache (`src/hf/cache/`), local bundled paths (`src/hf/local/`), or `package://` (Kraken).
New `hf://` entries should include both `hub_revision` and `artifact_sha256`; see
the migration note in [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

**Adding a model:** step-by-step checklist in [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

## Inference helper (local CPU on researcher machines)

For hosted SPA + local inference, run the slim helper sidecar (no Postgres, no job queue):

```bash
HELPER_REGISTRY_URL=http://localhost:8000/inference/v1/registry \
HF_CACHE_ROOT=~/.nomicous/hf/cache uv run --group inference python -m inference.helper
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/inference/v1/catalog
```

On startup the helper fetches `registry.yaml` from the hosted platform (`GET /inference/v1/registry`, public, ETag-aware) into `~/.nomicous/registry.yaml`. The bundled copy in the installer is only a fallback when offline. Local-eligible `hf://` weights are prefetched in a background thread on first launch so the first `/run` does not stall on download.

The browser probes `127.0.0.1:8001`, calls `/inference/v1/run`, and persists results through
the hosted platform API. The production Vercel CSP permits that exact loopback URL. Set an
explicit `HELPER_CORS_ORIGINS` runtime value only when the hosted SPA origin differs from
`https://app.nomicous.com`; do not use `*` or ship a helper secret in frontend code.

Packaging for `.dmg` / `.msi` / Linux installers: [`packaging/helper/README.md`](../packaging/helper/README.md) - PyInstaller spec excludes training stacks, platform API, and unused torch backends so installers ship only Calamari + Kraken runtime.

## Admission control and helper exposure

Both the API and local helper enforce the same `INFERENCE_*` limits before
base64 decoding or model loading. Defaults are: 160 MiB request body, 160 MiB encoded image,
100 MiB decoded image, 128 MiB job payload, 200 million pixels, 64 MiB parameters, depth 8,
8,000,000 parameter items, 10,000 transcription lines, 256 geometry points per line,
100 queued/running jobs, one worker thread, and 60 POSTs per minute per process. Allowed image formats default
to `JPEG,PNG,TIFF,WEBP`. Operators may lower or raise these with the corresponding
`INFERENCE_MAX_*`, `INFERENCE_WORKER_CONCURRENCY`, and
`INFERENCE_RATE_LIMIT_PER_MINUTE` environment variables; only trusted deployment
configuration should do so.

The API applies request-size and rate controls per process. Queue admission uses a PostgreSQL
transaction advisory lock, so the configured queue cap holds across API replicas. Worker
concurrency is bounded per worker process. It deliberately does **not** attempt to cancel
timed-out Python model threads: ML libraries cannot be safely killed in-process. Run workers
under a host/container supervisor with an execution deadline and restart policy; the existing
running-job timeout is a stale-lease recovery mechanism, not execution cancellation.

The helper defaults to `127.0.0.1` and is intentionally unauthenticated only in that loopback
mode. Binding it to any non-loopback host requires both `HELPER_SECURE_MODE=true` and a
non-placeholder `HELPER_AUTH_SECRET` of at least 32 characters. Secure mode protects all
helper routes with the `X-Inference-Helper-Secret` header. Put the helper behind TLS and an
authenticated reverse proxy when exposing it beyond the local machine.

## Tests

```bash
uv run --group inference pytest tests/inference tests/hf
```

Stop the Compose `inference-worker` before local integration runs (`docker stop nomicous-inference-worker-1`).
Full-suite layout, `DATABASE_URL` caveats, and failure analysis: [`docs/guides/testing.md`](../docs/guides/testing.md).

## Related docs

- Nomicous platform API and job integration: [`nomicous/backend/README.md`](../nomicous/backend/README.md)
- Compose stack and env vars: [`docker-compose.yml`](../docker-compose.yml) and [`nomicous/README.md`](../nomicous/README.md)
