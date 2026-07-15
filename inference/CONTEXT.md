# inference

Standalone inference service for manuscript **segment** and **transcribe** jobs. Loads models from a file-based **Registry** and resolves weight files at runtime.

## Language

**Registry**:
The YAML catalog (`registry.yaml`) that lists runnable models by **registry model id**, task, architecture, and weight location.
_Avoid_: Model catalog (ambiguous with platform Postgres catalog)

**Registry model id**:
The stable runtime key for one inference model (e.g. `greek-calamari-v1`). Format: `{script}-{architecture}-{model_version}`.
_Avoid_: model_id (ambiguous with platform UUID), Hub repo slug

**Hub repo slug**:
The single-segment name of a **Hub model repo**, derived from script and architecture: `{script}-htr-{architecture}` (e.g. `greek-htr-calamari`). The model card title may be human-friendly (e.g. "Ancient Greek HTR"); the slug stays mechanical.
_Avoid_: ancient-greek-htr (fine as display title, not slug), registry model id (includes model_version)

**Model version**:
The family generation of a model (`v1`, `v2`) - distinct from **registry tag** (`stable`). Encoded in **registry model id** and local staging path, not in **Hub repo slug** when architecture already disambiguates repos.
_Avoid_: version (too generic), release

**Script**:
The writing system / language family the model targets (e.g. `greek`, `syriac`). First segment of local staging path and **registry model id**.
_Avoid_: language (fine in metadata), locale

**Registry tag**:
A named version slot for one registry model id (e.g. `stable`). Selects which weight snapshot to load.
_Avoid_: version (too generic), release

**Weights source**:
A URI telling the inference service where to find checkpoint files for one registry tag. Schemes: local `file://`, packaged `package://`, or remote `hf://<namespace>/<hub-repo-slug>@<registry-tag>` (full Hub URI including namespace).
_Avoid_: artifact path, model path

**Hub model repo**:
A Hugging Face **model** repository holding inference weights and a model card for one **registry model id** (e.g. `nomicous/greek-htr-calamari` for `greek-calamari-v1`).
_Avoid_: model folder, checkpoint repo

**Hub dataset slug**:
The single-segment name of a **Hub dataset repo**, optimized for search: `{script}-manuscript-lines` or `{script}-{corpus}-htr-lines` (e.g. `greek-byzantine-manuscript-lines`). May carry a `nomos-` prefix for brand cohesion. Distinct from **Hub repo slug** (models use `{script}-htr-{architecture}`).
_Avoid_: mirroring registry model id, generic `dataset-v1`

**Hub collection**:
A Hugging Face collection grouping **Hub model repos** and **Hub dataset repos** for discovery. Source of truth: `src/hf/publish/collection.yaml`; synced via `scripts/hf/sync_collection.py`. Collection slug: `nomos`.
_Avoid_: monorepo, model bundle

**Hub revision**:
The immutable 40-character git commit on a **Hub model repo** selected by a **registry tag**. The tag remains a human-facing selector in the **weights source**, while the registry records its resolved commit separately.
_Avoid_: mutable tag as a runtime revision, version (too generic), release branch

**Hub artifact**:
The checkpoint files published inside a **Hub model repo** at one **Hub revision** - Calamari inference uses converted PyTorch `.pt` checkpoints; Kraken may use `.mlmodel` or `.safetensors`.
_Avoid_: weights (too generic), model file

**Artifact SHA-256**:
The required 64-character SHA-256 digest for the architecture-native **Hub artifact**. The inference service verifies it after download, before Hub-cache reuse, and before passing the artifact to an architecture loader.
_Avoid_: directory hash alone, unverified download

**Local bundled weights**:
Checkpoint files under `src/hf/local/` used for offline dev and Docker without Hub access. Referenced by `file://` **weights source** URIs relative to `src/hf/`.
_Avoid_: dev weights, inference/weights

**Hub staging tree**:
Publish-ready **Hub artifact**s under `src/hf/staging/` (models and datasets) before upload scripts push to the Hub.
_Avoid_: hf repo (ambiguous with Hub remote repo)

**Hub cache**:
Downloaded **Hub artifact**s at runtime under `src/hf/cache/<registry_model_id>/<registry_tag>/`. Reused only when required files exist and a manifest matches the immutable **Hub revision** and **artifact SHA-256**.
_Avoid_: runtime weight cache, inference/weights/cache

**Hub cache manifest**:
An integrity record (e.g. `.hub-manifest.json`) stored alongside cached **Hub artifact**s. It records the Hub repo, immutable **Hub revision**, artifact path, and **artifact SHA-256**; all must match before cache reuse.
_Avoid_: revision file alone (insufficient when artifact bytes change)

**Hub integration**:
Python code under `src/hf/` that resolves `hf://` URIs, checks **Hub cache**, and downloads missing artifacts. Used by `inference` at inference time and by `scripts/hf/`.
_Avoid_: huggingface module (too generic)

**Inference host**:
The machine where model weights are cached and inference executes - either the researcher's machine (**local inference**) or a hosted server (**remote inference**).
_Avoid_: Runtime (too generic), worker node (infra jargon)

**Lite model tier**:
A registry model id sized for CPU on a typical researcher laptop; eligible for **local inference** via **Inference helper**.
_Avoid_: Small model (vague), edge model (mobile jargon)

**Server model tier**:
A registry model id for **remote inference** only - too large or too slow for typical laptops.
_Avoid_: Large model (vague), cloud model (ambiguous with Hub hosting)

**Host eligibility**:
Whether a registry model id may run on the researcher's machine (`local`), only on a hosted server (`remote`), or either (`any`). Distinct from **Compute device** (`cpu` / `cuda`), which says what hardware to use once a host is chosen. All `local` models - transcribe and segment - run on **Inference helper** when it is present; cloud is fallback only.
_Avoid_: device (already means cpu/cuda), tier alone (ambiguous)

**Hub namespace**:
The Hugging Face account or org that owns **Hub model repos** and **Hub dataset repos**. Starts under a personal username; may later move to the `nomicous` org without changing repo slugs.
_Avoid_: org (when meaning the namespace generically)

## Relationships

- The **Registry** maps each **registry model id** + **registry tag** to one **weights source**
- One **Hub model repo** corresponds to one `{script}-htr-{architecture}` pair; **Hub repo slug** = `{script}-htr-{architecture}`
- **Registry model id** = `{script}-{architecture}-{model_version}`; maps to one **Hub repo slug** + **model version**
- Local **Hub staging tree**: `src/hf/staging/models/{script}/{architecture}/{model_version}/{registry_tag}/`
- **Hub cache**: `src/hf/cache/{registry_model_id}/{registry_tag}/`
- **Local bundled weights**: `src/hf/local/{script}/{architecture}/{model_version}/{registry_tag}/`
- All three live under `src/hf/` alongside **Hub integration** code
- **Hub cache** reuse requires matching **Hub cache manifest** hash, not just present files
- **Hub integration** lazy-fetches at inference; `scripts/hf/fetch_model.py` for explicit prefetch
- One **registry tag** records one immutable **Hub revision** on that repo
- Training output is copied into **Hub staging tree** when ready to publish
- One **Hub dataset repo** may train many **registry model ids** over time
- A **Hub collection** (`nomos`) links to many **Hub model repos** and **Hub dataset repos**; defined in `src/hf/publish/collection.yaml`

## Example dialogue

> **Dev:** "Should training crops live in the same HF repo as the Greek checkpoint?"
> **Domain expert:** "No. Weights go in a **Hub model repo**; labelled crops go in a **Hub dataset repo**. Inference only reads the Registry and **Hub model repos**."

## Flagged ambiguities

- "data" was used to mean both weights and training material - resolved: use **Hub model repo** vs **Hub dataset repo**.
- "kalamos" vs "nomicous" as public product name - resolved for Hub: product is **nomicous**; **Hub namespace** may be personal until the org exists.
- Checkpoint filename at repo root - resolved: use architecture-native names (**Hub artifact**), e.g. Calamari `best.pt`, not a forced `model.ckpt` rename.
- Calamari **Hub artifact** format is a converted PyTorch checkpoint (`.pt`); Kraken may use `.safetensors` or `.mlmodel`.
- Legacy registry ids (`greek-calamariv1`) - resolved: migrate to `{script}-{architecture}-{model_version}` (e.g. `greek-calamari-v1`); **Hub repo slug** uses hybrid `{script}-htr-{architecture}` pattern.
- **Hub cache** invalidation - resolved: manifest hash (not files-exist-only).

## Implementation notes (not domain)

### Current (interim)

- Sync inference: `POST /inference/v1/run` via `inference/api/run.py` and `inference/jobs/runner.py`
- Async jobs: `POST /inference/v1/jobs` plus `inference/jobs/worker.py` (Postgres queue, LISTEN/NOTIFY, callbacks)
- Architectures implemented: **Calamari** (`inference/architectures/calamari/`) and **Kraken segment** (`inference/architectures/kraken.py`)
- **Calamari runtime**: local PyTorch graph + local preprocessing; no TensorFlow or vendored `calamari_ocr` import at inference time.
- Weight resolution: `file://`, `hf://`, and `package://` (`src/hf/` local/cache layout; see `inference/weights/__init__.py`)
- Runtime cache: `src/hf/cache/<registry_model_id>/<registry_tag>/`

### Hub layout (`src/hf/`)

| Piece | Location |
|-------|----------|
| `hf://` resolution, download, cache manifest | `src/hf/resolve/` |
| Publish staging validation, model cards, collection sync | `src/hf/publish/` |
| Local bundled weights for offline dev | `src/hf/local/` |
| Publish-ready staging tree | `src/hf/staging/` |
| Hub runtime cache | `src/hf/cache/` |
| Collection metadata | `src/hf/publish/collection.yaml` |
| CLI entrypoints | `scripts/hf/` |
