# ML inference service

Standalone FastAPI service for manuscript **segment** and **transcribe** inference. It lives at the repository root in `inference/`, separate from the Nomicous platform API in `nomicous/backend/`.

For the public product overview, setup, model availability, and architecture,
see the [root README](../README.md), [use and hosting guide](../docs/guides/using-and-hosting.md),
[models and datasets guide](../docs/inference/models-and-datasets.md), and
[technical architecture](../docs/architecture.md).

## Status

| Piece | State |
|-------|--------|
| HTTP API (`inference/api/`) | Health and sync `/inference/v1/run` |
| Request/response contracts (`inference/contracts/`) | Defined for segment, transcribe, jobs, and callbacks |
| Model registry (`inference/registry.yaml`) | Calamari transcribe + BLLA segmentation entries |
| Model runner (`inference/jobs/runner.py`) | Registry lookup, weight resolution, and model execution |
| Local helper (`inference/helper/`) | Loopback sidecar serving the same sync run path |

## One queue, owned by the platform

This package holds no job queue, no database, and no claim loop. A queued page is
a row in the platform's `jobs` table; an inference agent claims it, runs it
through the same `run_model()` the sync path uses, and reports the outcome
through the platform's existing job callback contract. See
[ADR 0003](../docs/adr/0003-single-job-queue-cloud-worker-claims-like-a-device.md).

There is consequently no `inference-api` container: the registry endpoint an
agent syncs from is served by the platform on port 8000.

## Weights layout

Registry models resolve weights at runtime from:

| Source | Example | Cache / path |
|--------|---------|----------------|
| Hub | `hf://kkkamur07/syriac-htr-calamari@stable` | `src/hf/cache/<registry_model_id>/<registry_tag>/` |
| Local bundled (offline) | `file://local/syriac/calamari/v1/stable/best.pt` | `src/hf/local/...` |
| BLLA segmentation | `hf://kkkamur07/segmentation-blla@stable` | `blla.safetensors` in the Hub cache |

No local weight checkout is required for the default Hub models; they download from their public repos on first use into `HF_CACHE_ROOT`.

### Runtime

Both architectures run on PyTorch, CPU only ([ADR 0004](../docs/adr/0004-pytorch-is-the-inference-runtime.md)).
Transcribe loads a native `.pt` checkpoint through
`inference/architectures/calamari/`, and segment loads `blla.safetensors`
through `inference/architectures/blla/`. Every inference path calls
`model.eval()` and runs under `torch.inference_mode()`.

The **artifact SHA-256** is verified in `architectures/artifact.py` *before* the
architecture loader opens the file, which is what keeps `torch.load` (itself
called with `weights_only=True`) off an unverified checkpoint.

The retired ONNX Runtime path, its conversion scripts, and the parity harness
are in [`archive/onnx-runtime/`](../archive/onnx-runtime/README.md).

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
- `blla-segment` - segment, BLLA `safetensors` weights

Weights are resolved at runtime from Hub cache (`src/hf/cache/`) or local bundled paths (`src/hf/local/`).
New `hf://` entries should include both `hub_revision` and `artifact_sha256`; see
the migration note in [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

**Adding a model:** step-by-step checklist in [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

## Inference helper (local CPU on researcher machines)

For hosted SPA + local inference, run the slim helper sidecar (no Postgres, no job queue):

```bash
HELPER_REGISTRY_URL=http://localhost:8000/inference/v1/registry \
HF_CACHE_ROOT=~/.nomicous/hf/cache uv run --group inference python -m inference.helper
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/inference/v1/info
```

The helper serves three routes: `GET /health` (liveness, used by the installers),
`GET /inference/v1/info`, and `POST /inference/v1/run`.

`GET /inference/v1/info` is the single capability document and the only supported
discovery probe. Clients must check `service` before sending work: something else
may own port 8001, and a manuscript image should never be POSTed to it.

```json
{
  "service": "nomicous-inference-helper",
  "version": "0.1.6",
  "models": [
    {"registry_model_id": "blla-segment", "task": "segment",
     "host_eligibility": "local", "tags": ["stable"], "cached": true}
  ]
}
```

`cached` is a local-disk answer only: it means the pinned weights are present and
match their `artifact_sha256`, checked without contacting the Hub.

On startup the helper fetches `registry.yaml` from the hosted platform (`GET /inference/v1/registry`, public, ETag-aware) into `~/.nomicous/registry.yaml`. The bundled copy in the installer is only a fallback when offline. Model weights download lazily when the first `/run` needs them.

The browser calls `/inference/v1/run` through the configured helper URL, then falls back
to `127.0.0.1:8001` and `localhost:8001`. The production Vercel CSP permits
these loopback URLs. The helper accepts browser requests only from
`https://app.nomicous.com`.

Packaging for `.dmg` / `.msi` / Linux installers: [`packaging/helper/README.md`](../packaging/helper/README.md) - PyInstaller spec excludes training stacks and the platform API so installers ship only the Calamari + BLLA PyTorch CPU runtimes.

## Admission control and helper exposure

Both the API and local helper enforce the same `INFERENCE_*` limits before
base64 decoding or model loading. Defaults are: 160 MiB request body, 160 MiB encoded image,
100 MiB decoded image, 128 MiB job payload, 200 million pixels, 64 MiB parameters, depth 8,
8,000,000 parameter items, 10,000 transcription lines, 256 geometry points per line,
and 60 POSTs per minute per process. Allowed image formats default
to `JPEG,PNG,TIFF,WEBP`. Operators may lower or raise these with the corresponding
`INFERENCE_MAX_*` and `INFERENCE_RATE_LIMIT_PER_MINUTE` environment variables;
only trusted deployment configuration should do so.

These are per-process request controls only. Queue admission, rate limiting
across replicas, and stale-lease recovery belong to the platform's queue, which
is now the only one.

The helper is unauthenticated by design, so the loopback bind is what keeps it off the
network. `HELPER_HOST` must be a loopback address; any other value is rejected at startup,
and the listening socket is IPv4 `127.0.0.1` only. Exposing the helper to other machines is
not supported: run inference on the hosted API instead.

## Tests

```bash
uv run --group inference --group export pytest tests/inference tests/hf
```

The slow native-BLLA parity suite installs the original Kraken implementation
only through the development-only `parity` group:

```bash
uv run --group inference --group export --group parity pytest tests/inference/integration/test_blla_parity.py -q
```

Full-suite layout, `DATABASE_URL` caveats, and failure analysis: [`docs/guides/testing.md`](../docs/guides/testing.md).

## Related docs

- Nomicous platform API and job integration: [`nomicous/backend/README.md`](../nomicous/backend/README.md)
- Compose stack and env vars: [`docker-compose.yml`](../docker-compose.yml) and [`nomicous/README.md`](../nomicous/README.md)
