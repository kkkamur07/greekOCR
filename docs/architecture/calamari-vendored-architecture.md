# Calamari vendored architecture (migration off the PyPI library)

## Summary

Greek and Syriac **transcribe** jobs use the Calamari HTR stack. We no longer rely on the **installed `calamari-ocr` Python package** as the source of `calamari_ocr` at inference time. The repo carries a **vendored copy** of the Calamari architecture and predictor code; inference loads that copy from disk.

**Kraken** (segment) is unchanged — it remains a normal pip dependency (`kraken`).

| Concern | Before | Now |
|---------|--------|-----|
| `calamari_ocr` import source | PyPI `calamari-ocr` and/or local `_support_repo/calamari` | **Vendored tree** (`src/model/calamari`) |
| Canonical source in git | `_support_repo/calamari` (gitignored) | **`src/model/calamari`** (committed) |
| Docker inference image | Often missing Calamari source → transcribe failed | **`COPY src/model/calamari` → `/app/_support_repo/calamari`** |
| Training | `src/train/calamari/` + vendored model code | Same vendored tree |

We still install **TensorFlow and related wheels** from the `inference` dependency group in `pyproject.toml`. The migration is about **owning the Calamari Python architecture**, not about removing TensorFlow yet.

---

## Why we vendored

1. **Reproducibility** — Checkpoints embed Calamari-specific class paths and SavedModel metadata. A pinned vendored tree matches our trained `best.ckpt` artifacts.
2. **No upstream surprise** — PyPI `calamari-ocr` releases can diverge from the fork we trained against. Vendoring freezes the code we control.
3. **Docker and CI** — `_support_repo/` was gitignored and excluded from `.dockerignore`, so inference containers built without Calamari source failed transcribe jobs with:

   ```text
   CalamariUnavailableError: local Calamari architecture source not found: /app/_support_repo/calamari
   ```

4. **Path to a slimmer runtime** — Longer term we intend to port the **model and forward pass** into a first-class `inference/architectures/` implementation (see [Future work](#future-work)) so we are not tied to the full Calamari + TensorFlow stack for every platform (especially slow `linux/amd64` emulation on Apple Silicon).

---

## Layout on disk

```text
src/model/calamari/          # Canonical vendored Calamari (in git)
  calamari_ocr/              # Python package imported at runtime
    ocr/
      dataset/               # Required for predict (Codec, pipelines) — must not be omitted
      predict/predictor.py
    resources/               # Rulesets used by text processors
    ...

_support_repo/calamari/      # Legacy runtime path (optional on host; gitignored)
  calamari_ocr/              # Often a copy or symlink of src/model/calamari for local dev

inference/
  architectures/calamari.py  # Adapter: loads predictor, maps to TranscribeRunResponse
  weights/calamari/            # Checkpoints (mounted in Compose; not in image layer)
    greek-calamariv1/
    syriac-calamariv1/

inference/Dockerfile           # COPY src/model/calamari → /app/_support_repo/calamari
```

### Runtime path resolution

`inference/architectures/calamari.py` resolves the repo root and expects vendored code here:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]   # repository root (/app in Docker)
SUPPORT_CALAMARI_ROOT = REPO_ROOT / "_support_repo" / "calamari"
```

On import it prepends that directory to `sys.path` and imports:

```python
from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
```

The directory name `_support_repo` is **historical**. In Docker and fresh clones, populate it from **`src/model/calamari`**, not from PyPI.

---

## What still comes from pip

The `inference` group in `pyproject.toml` installs runtime libraries the vendored code needs:

- `tensorflow` (and `tensorflow-macos` on Darwin)
- `ocrd-fork-tfaip`, `h5py`, `python-bidi`, `edit-distance`, …
- `calamari-ocr` is still listed for dependency resolution / tooling compatibility, but **inference must not depend on site-packages `calamari_ocr` being the loaded module** — the vendored path is inserted first on `sys.path`.

Platform API containers (`nomicous-api`) do **not** include Calamari or TensorFlow; only `inference-api` and `inference-worker` run transcribe.

---

## Docker Compose

`inference/Dockerfile` runtime stage:

```dockerfile
COPY inference /app/inference
COPY src/model/calamari /app/_support_repo/calamari
```

`.dockerignore` excludes `_support_repo/` (host-only convenience). The image **must** copy from `src/model/calamari` so builds are reproducible without a local `_support_repo` checkout.

Weights are **not** baked into the image; Compose mounts them:

```yaml
volumes:
  - ./inference/weights:/app/inference/weights
```

After changing the Dockerfile or vendored source:

```bash
docker compose build inference-api
docker compose up -d inference-api inference-worker
```

Verify inside the container:

```bash
docker compose run --rm --no-deps inference-worker \
  python -c "from pathlib import Path; p=Path('/app/_support_repo/calamari'); print(p, p.is_dir())"
```

Expected: `/app/_support_repo/calamari True`

---

## Local development (without Docker)

From the repository root:

```bash
uv sync --group inference
```

Ensure vendored Calamari is visible at the legacy path (one-time per clone if you do not use Docker):

```bash
mkdir -p _support_repo
# If _support_repo/calamari is missing, symlink or copy:
ln -sfn ../src/model/calamari _support_repo/calamari
```

Run inference API or worker with `PYTHONPATH=.`:

```bash
PYTHONPATH=. uvicorn inference.api.main:app --host 0.0.0.0 --port 8001
PYTHONPATH=. python -m inference.jobs.worker
```

Training entry points under `src/train/calamari/` use the same `src/model/calamari` tree.

---

## Registry and checkpoints

`inference/registry.yaml` registers Calamari transcribe models, for example:

| Registry id | Weights (interim) |
|-------------|-------------------|
| `greek-calamariv1` | `file://weights/calamari/greek-calamariv1/stable.ckpt` |
| `syriac-calamariv1` | `file://weights/calamari/syriac-calamariv1/best.ckpt` |

The adapter in `inference/architectures/calamari.py` loads the checkpoint via vendored `Predictor.from_checkpoint`. Missing vendored source fails **before** checkpoint IO with `CalamariUnavailableError`. Missing weights fail with `FileNotFoundError`.

---

## Job failure symptoms

| Symptom | Cause |
|---------|--------|
| `CalamariUnavailableError: local Calamari architecture source not found` | `/app/_support_repo/calamari` absent in inference image or host layout |
| `ModuleNotFoundError: No module named 'calamari_ocr.ocr.dataset'` | **Incomplete vendored tree** — `src/model/calamari/calamari_ocr/ocr/dataset/` missing; sync from full Calamari fork (see layout above) |
| `CalamariUnavailableError: … install the project with the calamari extra` | Vendored path exists but import failed (often missing `dataset/` or other subpackages) |
| UI shows **Inference job failed** (generic) | Worker maps exceptions to a public message; see `inference-worker` logs for the traceback |
| Segmentation works, OCR fails with `CalamariUnavailableError` | Kraken is pip-only; OCR needs vendored Calamari + TensorFlow in **inference-worker** |
| Job **done** but UI shows no model text | OCR ran but returned **empty** `text` (often **Syriac model on Greek pages**). Check `jobs.result` and `line_transcriptions`. Use `greek-calamariv1` for Byzantine Greek seed data — weights must exist under `inference/weights/calamari/greek-calamariv1/` (see below). |
| `greek-calamariv1` job fails with checkpoint not found | Copy trained Greek checkpoint to `inference/weights/calamari/greek-calamariv1/stable.ckpt/` (+ sidecar JSON). Only `syriac-calamariv1` weights ship in-repo today. |
| Very slow first OCR | TensorFlow cold start + `platform: linux/amd64` on ARM Macs (Compose comment) |

---

## Relationship to training (`src/`)

| Path | Role |
|------|------|
| `src/model/calamari/` | Vendored Calamari library (single source of truth in git) |
| `src/train/calamari/` | Hydra training / finetune scripts |
| `configs/calamari_*.yaml` | Training presets |
| `outputs/` | Training checkpoints (copy into `inference/weights/calamari/` for inference) |

Root [`README.md`](../../README.md) documents recognition quality and training commands.

---

## Future work

Tracked intent (see also [`todo.md`](../todo.md)):

1. **Extract a minimal inference architecture** — Reimplement or wrap only the forward pass and checkpoint loader under `inference/architectures/`, reducing dependence on the full vendored Calamari tree.
2. **TensorFlow / platform matrix** — Smaller inference images and faster ARM builds once transcribe no longer requires the full TF stack in the default path.
3. **Hub weights** — Move checkpoints from `inference/weights/` to `src/hf/` layout described in [`inference/CONTEXT.md`](../../inference/CONTEXT.md) (planned, not fully implemented).

Until (1) ships, treat **`src/model/calamari` + Docker COPY** as the supported Calamari integration.

---

## Related docs

- [`inference/README.md`](../../inference/README.md) — inference service, Compose, registry
- [`inference/CONTEXT.md`](../../inference/CONTEXT.md) — domain terms, Hub target layout
- [`inference/architectures/calamari.py`](../../inference/architectures/calamari.py) — adapter implementation
- [`inference/Dockerfile`](../../inference/Dockerfile) — image build including vendored Calamari
- [`docs/decisions/001-platform-job-status-push.md`](../decisions/001-platform-job-status-push.md) — job polling vs push (separate from Calamari)
