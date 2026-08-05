# inference

Standalone inference service for manuscript **segment** and **transcribe** jobs. Loads models from a file-based **Registry** and resolves weight files at runtime.

## Language

**Registry**:
The YAML catalog (`registry.yaml`) that lists runnable models by **registry model id**, task, architecture, and weight location.
_Avoid_: Model catalog (ambiguous with platform Postgres catalog)

**Registry model id**:
The stable runtime key for one inference model (e.g. `greek-calamari-v1`).
Language-specific format: `{script}-{architecture}-{model_version}`;
script-agnostic models may use a task-specific id such as `blla-segment`.
_Avoid_: model_id (ambiguous with platform UUID), Hub repo slug

**Hub repo slug**:
The single-segment name of a **Hub model repo**, derived from task and architecture: HTR repos use `{script}-htr-{architecture}` (e.g. `greek-htr-calamari`), while the script-agnostic BLLA segmenter uses `segmentation-blla`. The model card title may be human-friendly; the slug stays mechanical.
_Avoid_: ancient-greek-htr (fine as display title, not slug), registry model id (includes model_version)

**Model version**:
The family generation of a model (`v1`, `v2`) - distinct from **registry tag** (`stable`). Encoded in **registry model id** and local staging path, not in **Hub repo slug** when architecture already disambiguates repos.
_Avoid_: version (too generic), release

**Script**:
The writing system / language family the model targets (e.g. `greek`, `syriac`). Script-agnostic models use a reserved task namespace such as `segmentation`. It is the first segment of the local staging path and **registry model id**.
_Avoid_: language (fine in metadata), locale

**Registry tag**:
A named version slot for one registry model id (e.g. `stable`). Selects which weight snapshot to load.
_Avoid_: version (too generic), release

**Weights source**:
A URI telling the inference service where to find checkpoint files for one registry tag. Schemes: local `file://`, packaged `package://`, or remote `hf://<namespace>/<hub-repo-slug>@<registry-tag>` (full Hub URI including namespace).
_Avoid_: artifact path, model path

**Hub model repo**:
A Hugging Face **model** repository holding inference weights and a model card for one **registry model id** (e.g. `nomicous/greek-htr-calamari` for `greek-calamari-v1` or `nomicous/segmentation-blla` for `blla-segment`).
_Avoid_: model folder, checkpoint repo

**Hub dataset slug**:
The single-segment name of a **Hub dataset repo**, optimized for search: `{script}-manuscript-lines` or `{script}-{corpus}-htr-lines` (e.g. `greek-byzantine-manuscript-lines`). May carry a `nomos-` prefix for brand cohesion. Distinct from **Hub repo slug** (models use task-specific slugs).
_Avoid_: mirroring registry model id, generic `dataset-v1`

**Hub collection**:
A Hugging Face collection grouping **Hub model repos** and **Hub dataset repos** for discovery. Source of truth: `src/hf/publish/collection.yaml`; synced via `scripts/hf/sync_collection.py`. Collection slug: `nomos`.
_Avoid_: monorepo, model bundle

**Hub revision**:
The immutable 40-character git commit on a **Hub model repo** selected by a **registry tag**. The tag remains a human-facing selector in the **weights source**, while the registry records its resolved commit separately.
_Avoid_: mutable tag as a runtime revision, version (too generic), release branch

**Hub artifact**:
The checkpoint files published inside a **Hub model repo** at one **Hub revision** - Calamari and BLLA inference load native PyTorch checkpoints (`.pt`, `.safetensors`) directly. There is no conversion step between the trained artifact and the run artifact.
_Avoid_: weights (too generic), model file, ONNX artifact (retired runtime)

**Artifact SHA-256**:
The required 64-character SHA-256 digest for the architecture-native **Hub artifact**. The inference service verifies it after download, before Hub-cache reuse, and before passing the artifact to an architecture loader.
_Avoid_: directory hash alone, unverified download

**Local bundled weights**:
Checkpoint files under `src/hf/local/` used for offline dev and Docker without Hub access. Referenced by `file://local/...` **weights source** URIs relative to `src/hf/`. A source-checkout affordance only: they are not shipped in the published package, which resolves weights by `hf://`. Override the root with `NOMICOUS_LOCAL_WEIGHTS_ROOT`.
_Avoid_: dev weights, inference/weights

**Hub staging tree**:
Publish-ready **Hub artifact**s under `src/hf/staging/` (models and datasets) before upload scripts push to the Hub.
_Avoid_: hf repo (ambiguous with Hub remote repo)

**Hub cache**:
Downloaded **Hub artifact**s at runtime under `~/.nomicous/hf/cache/<registry_model_id>/<registry_tag>/` (override: `HF_CACHE_ROOT`). It lives in the researcher's home directory, not beside the code, because **Hub integration** ships inside the installed package. Reused only when required files exist and a manifest matches the immutable **Hub revision** and **artifact SHA-256**.
_Avoid_: runtime weight cache, inference/weights/cache, src/hf/cache (pre-package layout)

**Hub cache manifest**:
An integrity record (e.g. `.hub-manifest.json`) stored alongside cached **Hub artifact**s. It records the Hub repo, immutable **Hub revision**, artifact path, and **artifact SHA-256**; all must match before cache reuse.
_Avoid_: revision file alone (insufficient when artifact bytes change)

**Hub integration**:
Python code at `inference/hub/` that resolves `hf://` URIs, checks **Hub cache**, verifies **artifact SHA-256**, and downloads missing artifacts. It lives inside the published package because it is on the runtime path (ADR 0002). Publish-side code under `src/hf/` reuses it; the reverse dependency does not exist.
_Avoid_: huggingface module (too generic), src/hf/resolve (pre-package location)

**Published package**:
The one distribution, `nomicous-inference`, carrying the inference library, **Hub integration**, and the `nomicous` console entry point. A researcher's laptop and a hosted worker install the same package (ADR 0002), so there is no version-compatibility matrix between components that always ship together. Built from the repository root; the wheel contains `inference/` minus the loopback HTTP surfaces.
_Avoid_: helper bundle (frozen-installer era), library package vs CLI package (there is one)

**Inference host**:
The machine where model weights are cached and inference executes - either the researcher's machine (**local inference**) or a hosted server (**remote inference**).
_Avoid_: Runtime (too generic), worker node (infra jargon)

**Lite model tier**:
A registry model id sized for CPU on a typical researcher laptop; eligible for **local inference**.
_Avoid_: Small model (vague), edge model (mobile jargon)

**Server model tier**:
A registry model id for **remote inference** only - too large or too slow for typical laptops.
_Avoid_: Large model (vague), cloud model (ambiguous with Hub hosting)

**Host eligibility**:
Whether a registry model id may run on the researcher's machine (`local`), only on a hosted server (`remote`), or either (`any`). Distinct from **Compute device** (`cpu` / `cuda`), which says what hardware to use once a host is chosen. Constrains which **execution target**s a job may choose; it does not choose one.
_Avoid_: device (already means cpu/cuda), tier alone (ambiguous)

**Execution target**:
The **inference host** a single job runs on - `local` or `cloud` - fixed when the job is submitted and never changed afterwards. A target may only be chosen when it has **capacity**; the researcher is told which host will run the job, and a job that fails reports which host it failed on.
_Avoid_: routing/preference (implies the platform may re-decide mid-flight), fallback (nothing changes target after submission)

**Capacity**:
Whether an **inference host** currently has a machine able to take work, answered by whether any device for that host was seen recently. The researcher's laptop and a hosted worker are the same kind of thing here, so cloud availability is not a separate concept. "Recently" is the device layer's existing idle window; a device with no **capacity** is not a failure, it is an announced state.
_Avoid_: helper available (loopback-era, meant a port answering), online (ambiguous with the researcher's own connectivity)

**Inference agent**:
The program that takes work from the platform and runs it - one implementation, run either on a researcher's machine (**local inference**) or on a hosted server (**remote inference**). Local and cloud differ by credential and uptime, not by code path.
_Avoid_: helper (loopback-era, meant a process listening on a port), worker (ambiguous with the platform's own job worker)

**Claim**:
One **inference agent** taking exactly one page of work from the platform's queue, over HTTP, authenticated as itself. A batch is N claims. The claim fixes nothing new: it hands over a job whose **execution target** was already decided at submission, and it may only hand over work for the target the presented credential is allowed to run.
_Avoid_: dispatch (implies the platform pushes), assignment (implies the platform chooses an agent), reservation

**Lease**:
How long a claimed page stays with the **inference agent** that took it before the platform may give it to another. There is no heartbeat: work is seconds-to-minutes, so the lease covers it with margin, and a stopped agent loses one page rather than a document.
_Avoid_: timeout (ambiguous with the platform-wide job timeout), lock (nothing is held in the database between requests)

**Service credential**:
The credential a hosted **inference agent** presents instead of a device token. It claims `cloud` work for the whole platform, which is why it is not a device token: a device token's entire authorization scope is the one account on its device row, and `cloud` work has no such owner. Its **capacity** row is owned by a service account no person can log into.
_Avoid_: device token for cloud, webhook secret (that authenticates the platform's own callback receiver), API key

**Host preference**:
The account-level setting "use my computer when it is available", the only researcher input to **execution target** selection. Combined with **host eligibility** and **capacity** it fixes one target at submission; there is no per-job toggle, because a researcher cannot know which host is faster for a given page.
_Avoid_: per-job execution mode, `local_only` (retired by ADR 0002), routing rule

**Hub namespace**:
The Hugging Face account or org that owns **Hub model repos** and **Hub dataset repos**. Starts under a personal username; may later move to the `nomicous` org without changing repo slugs.
_Avoid_: org (when meaning the namespace generically)

## Relationships

- The **Registry** maps each **registry model id** + **registry tag** to one **weights source**
- One **Hub model repo** corresponds to one task/architecture pair; HTR uses `{script}-htr-{architecture}` and BLLA segmentation uses `segmentation-blla`
- **Registry model id** = `{script}-{architecture}-{model_version}`; maps to one **Hub repo slug** + **model version**
- Local **Hub staging tree**: `src/hf/staging/models/{script}/{architecture}/{model_version}/{registry_tag}/`
- **Hub cache**: `~/.nomicous/hf/cache/{registry_model_id}/{registry_tag}/`
- **Local bundled weights**: `src/hf/local/{script}/{architecture}/{model_version}/{registry_tag}/`
- The **Hub staging tree** and **Local bundled weights** live under `src/hf/`; the **Hub cache** is in the researcher's home directory and **Hub integration** code is in the package
- **Hub cache** reuse requires matching **Hub cache manifest** hash, not just present files
- **Hub integration** lazy-fetches at inference; `scripts/hf/fetch_model.py` for explicit prefetch
- One **registry tag** records one immutable **Hub revision** on that repo
- Training output is copied into **Hub staging tree** when ready to publish
- One **Hub dataset repo** may train many **registry model ids** over time
- **Host eligibility** constrains which **execution target**s a job may choose; **host preference** and **capacity** choose one from what is left
- **Capacity** for one **inference host** = any device recorded as that host was seen recently; a hosted worker is such a device, not a separate concept
- One **claim** = one page; the presented credential fixes the **execution target** it may take (device token -> `local`, own account only; **service credential** -> `cloud`, any account)
- A **claim** starts a **lease**; the page returns to the queue when the lease expires, and completion or failure ends it through the platform's existing job callback contract
- A **Hub collection** (`nomos`) links to many **Hub model repos** and **Hub dataset repos**; defined in `src/hf/publish/collection.yaml`

## Example dialogue

> **Dev:** "Should training crops live in the same HF repo as the Greek checkpoint?"
> **Domain expert:** "No. Weights go in a **Hub model repo**; labelled crops go in a **Hub dataset repo**. Inference only reads the Registry and **Hub model repos**."

## Flagged ambiguities

- "data" was used to mean both weights and training material - resolved: use **Hub model repo** vs **Hub dataset repo**.
- "kalamos" vs "nomicous" as public product name - resolved for Hub: product is **nomicous**; **Hub namespace** may be personal until the org exists.
- Checkpoint filename at repo root - resolved: use native runtime artifact names (**Hub artifact**), e.g. Calamari `best.pt` and BLLA `blla.safetensors`.
- Calamari and BLLA runtime **Hub artifact** format is native PyTorch (`.pt`, `.safetensors`). ONNX was the runtime until 2026-08-04 and is retired - the trained artifact and the run artifact are now the same file, which is what removed the parity problem. See ADR 0004.
- Legacy registry ids (`greek-calamariv1`) - resolved: migrate to `{script}-{architecture}-{model_version}` (e.g. `greek-calamari-v1`); **Hub repo slug** is task-specific.
- **Hub cache** invalidation - resolved: manifest hash (not files-exist-only).

## Implementation notes (not domain)

### Current (interim)

- Sync inference: `POST /inference/v1/run` via `inference/api/run.py` and `inference/jobs/runner.py`
- Queued jobs: the platform owns the only queue (ADR 0003). `inference` holds no database, ORM, or claim loop of its own.
- **Claim**: `POST /device/v1/jobs/claim` on the platform - the one endpoint this layer adds (ADR 0005). Completion and failure are the existing `POST /internal/inference/job-complete` with a `JobCallbackRequest`; abandonment is the existing stale sweep. There is no heartbeat and no release endpoint.
- Architectures implemented: **Calamari** (`inference/architectures/calamari/`) and native **BLLA segment** (`inference/architectures/blla/`)
- **Calamari runtime**: local PyTorch graph + local preprocessing; no TensorFlow or vendored `calamari_ocr` import at inference time.
- Weight resolution: `file://`, `hf://`, and `package://` (see `inference/weights/__init__.py` and `inference/hub/`)
- Runtime cache: `~/.nomicous/hf/cache/<registry_model_id>/<registry_tag>/`

### Hub layout (`src/hf/`)

| Piece | Location |
|-------|----------|
| `hf://` resolution, download, cache manifest | `inference/hub/` (published package) |
| Publish staging validation, model cards, collection sync | `src/hf/publish/` |
| Local bundled weights for offline dev | `src/hf/local/` |
| Publish-ready staging tree | `src/hf/staging/` |
| Hub runtime cache | `~/.nomicous/hf/cache/` |
| Collection metadata | `src/hf/publish/collection.yaml` |
| CLI entrypoints | `scripts/hf/` |
