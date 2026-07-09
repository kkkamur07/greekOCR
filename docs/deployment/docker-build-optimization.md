# Docker build optimization

Review notes from the July 2026 pass to make Docker builds **faster** and images **smaller**.

## Summary

| Image | Before | After | Change |
|-------|--------|-------|--------|
| `nomicous-api` | 882–907 MB | **564 MB** | ~38% smaller |
| `nomicous-frontend` (production) | 550 MB | **77 MB** | ~86% smaller |
| `nomicous-inference` | 1.91 GB | 1.92 GB | ~unchanged (PyTorch-dominated) |
| Build context (repo root) | ~3 GB | **~60 MB** | ~98% smaller |

The largest wins came from **not sending local data into the build context** and **multi-staging the platform API image** so compilers and dev tools never land in the runtime layer.

---

## What was consuming resources

### 1. Build context (~3 GB)

Docker sends the entire build context to the daemon before any `Dockerfile` instruction runs. These paths were included but never needed inside images:

| Path | Size | Why it was unnecessary |
|------|------|------------------------|
| `data/` | ~1.9 GB | Local manuscript fixtures; mounted at runtime in compose |
| `packaging/` | ~1.0 GB | PyInstaller helper packaging tree; not used by API/inference images |
| `nomicous/frontend/` | ~328 MB | Built from a separate `nomicous/` context |
| `nomicous/backend/media/` | ~88 MB | User uploads; volume-mounted in compose |
| `tests/`, `docs/`, `issues/` | small | Dev-only |

**Fix:** expanded [`.dockerignore`](../.dockerignore).

### 2. Platform API image (`nomicous/Dockerfile`)

The API image was a **single-stage** build that:

- Installed `gcc` and `libpq-dev` (~191 MB apt layer) into the final image
- Copied the `uv` binary (~53 MB) into runtime even though only `.venv` is needed
- Installed `pytest` / `pytest-asyncio` via the `platform` dependency group
- Copied the full `inference/` tree when the API only imports `inference.contracts`

None of the platform Python deps require compiling C extensions at image build time (`psycopg2-binary`, `asyncpg`, etc. ship wheels).

### 3. Frontend production image (`nomicous/frontend/Dockerfile`)

The `runner` stage used `node:20-alpine` plus a global `serve@14` install. That is convenient but heavy for serving static files.

### 4. Inference image (`inference/Dockerfile`)

Already multi-stage. The `.venv` layer is ~1.31 GB — almost entirely **PyTorch + Kraken**. There is little room to shrink without changing the ML stack or publishing a pre-built base image with those deps cached in a registry.

---

## Changes made

### `.dockerignore` (repo root)

Excluded from root-context builds (`api`, `inference-api`, `inference-worker`):

```
data/
packaging/
tests/
docs/
configs/
inference/weights
src/**          # inference only needs src/hf/
!src/hf/
nomicous/frontend/
nomicous/backend/media
nomicous/data/
nomicous/backend/tests
```

`nomicous/.dockerignore` was also tightened for frontend-only builds (excludes `backend/`, `infrastructure/`, etc.).

### `nomicous/Dockerfile` — multi-stage API

**Before:** single stage with apt build tools + `uv` + full `inference/`.

**After:**

1. **`builder`** — `uv sync --frozen --only-group platform-prod`
2. **`runtime`** — copy `.venv` only; install `fonts-dejavu-core` (ReportLab PDFs); copy app code
3. Copy only `inference/__init__.py` and `inference/contracts/` (platform imports contracts only)

Runtime no longer contains `gcc`, `libpq-dev`, or the `uv` binary.

### `pyproject.toml` — `platform-prod` group

New dependency group for production API images — same as `platform` but **without** `pytest` and `pytest-asyncio`.

| Group | Used by |
|-------|---------|
| `platform` | Local dev and tests (`uv sync --group platform`) |
| `platform-prod` | `nomicous/Dockerfile` |

### `inference/Dockerfile`

- Strip `__pycache__` / `.pyc` from `.venv` after `uv sync`
- Add `libgomp1` in runtime (OpenMP for PyTorch on slim images)

### `nomicous/frontend/Dockerfile` — production runner

| Stage | Purpose | Base image |
|-------|---------|------------|
| `dev` | `docker compose` local dev (Vite HMR) | `node:20-alpine` |
| `builder` | `npm ci` + `npm run build` | `node:20-alpine` |
| `runner` | Serve static `dist/` | `nginx:1.27-alpine` |

Added [`nomicous/frontend/nginx.conf`](../nomicous/frontend/nginx.conf) — SPA fallback on port 5173 to match compose.

### Minor fix

`nomicous/frontend/src/utils/jobPolling.ts` — prefixed unused `jobId` param with `_` so `tsc` passes in the production frontend build.

---

## Build commands

### Development (unchanged)

From repo root:

```bash
docker compose up --build
```

Compose still builds the frontend with `target: dev` (Vite hot reload).

### Production-style images

```bash
# Platform API
docker build -f nomicous/Dockerfile -t nomicous-api .

# Inference (API + worker share this image)
docker build -f inference/Dockerfile -t nomicous-inference .

# Frontend static assets
docker build -f nomicous/frontend/Dockerfile --target runner -t nomicous-frontend ./nomicous
```

### Measure build context size locally

```bash
tar -cf - --exclude-from=.dockerignore . | wc -c | awk '{printf "%.1f MB\n", $1/1024/1024}'
```

---

## Image layer breakdown (API, after optimization)

| Layer | Approx. size | Notes |
|-------|--------------|-------|
| `python:3.11-slim` base | ~150 MB | Unchanged |
| `.venv` (platform-prod) | ~262 MB | Down from ~269 MB; no pytest |
| App source | ~1 MB | backend + infrastructure + contracts |
| `fonts-dejavu-core` | ~15 MB | PDF generation |
| ~~`gcc` + `libpq-dev`~~ | ~~191 MB~~ | **Removed from runtime** |
| ~~`uv` binary~~ | ~~53 MB~~ | **Builder only** |

---

## What was not changed (and why)

| Item | Reason |
|------|--------|
| **Inference image size** | PyTorch + Kraken dominate; requires architectural change to meaningfully reduce |
| **Compose frontend `dev` target** | Local dev needs Vite HMR, not nginx |
| **Pooler / separate DB roles** | Explicitly deferred in the production-security work |
| **Pre-built ML base image** | Possible follow-up: publish `nomicous-inference-base` with torch pre-installed to skip repeated downloads in CI |

---

## Follow-up ideas (not implemented)

1. **CI base image for inference** — cache torch/kraken in a registry image; inference `Dockerfile` becomes `FROM nomicous-inference-base` + copy code. Biggest potential CI time win.
2. **Docker Buildx bake** — single `docker-bake.hcl` for api / inference / frontend with shared cache mounts.
3. **Split inference API vs worker images** — worker needs full ML stack; a thin API image might only need FastAPI + job queue glue (larger refactor).
4. **Pin nginx and uv digests** — reproducible builds in CI.

---

## Files touched

| File | Change |
|------|--------|
| [`.dockerignore`](../.dockerignore) | Exclude large local trees from root context |
| [`nomicous/.dockerignore`](../nomicous/.dockerignore) | Exclude backend/infra from frontend context |
| [`nomicous/Dockerfile`](../nomicous/Dockerfile) | Multi-stage; `platform-prod`; contracts-only |
| [`inference/Dockerfile`](../inference/Dockerfile) | venv cleanup; `libgomp1` |
| [`nomicous/frontend/Dockerfile`](../nomicous/frontend/Dockerfile) | nginx `runner` stage |
| [`nomicous/frontend/nginx.conf`](../nomicous/frontend/nginx.conf) | Static SPA config |
| [`pyproject.toml`](../pyproject.toml) | `platform-prod` dependency group |
| [`uv.lock`](../uv.lock) | Regenerated for new group |
| [`nomicous/frontend/src/utils/jobPolling.ts`](../nomicous/frontend/src/utils/jobPolling.ts) | TS build fix |

---

## Verification (July 2026)

Built with `DOCKER_BUILDKIT=1` on Apple Silicon (linux/arm64):

```bash
docker build --no-cache -f nomicous/Dockerfile -t nomicous-api:opt .
docker build --no-cache -f inference/Dockerfile -t nomicous-inference:opt .
docker build --no-cache -f nomicous/frontend/Dockerfile --target runner -t nomicous-frontend:opt ./nomicous
```

Results:

```
nomicous-api:opt         564 MB   (was 882–907 MB)
nomicous-frontend:opt     77 MB   (was 550 MB)
nomicous-inference:opt  1.92 GB   (was 1.91 GB)
```

API cold build (no cache): ~30 s wall time, dominated by `uv sync` (~10 s) and layer export (~13 s). Inference cold build: ~2.5–3 min, dominated by torch wheel install/export.
