# Hugging Face Hub Integration — Product Requirements Document

> **Status: completed** for runtime pull, publish, collection, and registry migration (issues 030–036). Remaining slice: issue **034** (dataset staging publish tests/docs).

## Problem Statement

Inference models (Kraken segment, Calamari transcribe) today ship as local `file://` or `package://` **weights sources** under `inference/weights/`. That works for a single developer machine or a Docker image with bundled checkpoints, but it does not scale to:

- Fresh deployments that should pull the same **registry tag** from a remote store without copying multi‑gigabyte artifacts into git or every image layer.
- Collaborators or CI who need identical model versions without manual weight handoffs.
- Publishing new model generations after training without redeploying the whole inference stack.
- Discovering which **Hub model repos** and **Hub dataset repos** belong to the nomicous project.

Researchers and operators should be able to run segment and transcribe jobs against models resolved from the **Registry**, whether weights are bundled locally for offline dev or fetched lazily from a **Hub model repo** at runtime.

## Solution

Implement **Hub integration** as described in `inference/CONTEXT.md`: a dedicated module that resolves `hf://<namespace>/<hub-repo-slug>@<registry-tag>` **weights sources**, maintains a **Hub cache** with **Hub cache manifest** integrity checks, supports **local bundled weights** for offline Docker and dev, and provides publish/sync scripts for **Hub staging tree** → Hub upload.

The inference service keeps its existing HTTP and job contracts; only weight resolution and on-disk layout change. The platform Postgres **InferenceModel** catalog continues to reference models via `registry://<registry_model_id>?tag=<tag>` — Hub is an implementation detail behind the inference **Registry**.

## User Stories

### Runtime model pull

1. As an inference operator, I want the **Registry** to accept `hf://` **weights sources**, so that models can be loaded from **Hub model repos** without baking every checkpoint into the image.
2. As an inference operator, I want the service to lazy-download missing **Hub artifacts** on first use, so that cold starts do not require a separate manual prefetch step.
3. As an inference operator, I want an explicit prefetch command for a **registry model id** + **registry tag**, so that I can warm the **Hub cache** before serving traffic.
4. As an inference operator, I want the **Hub cache** to reuse artifacts when the **Hub revision** has not changed, so that restarts do not re-download large checkpoints.
5. As an inference operator, I want the **Hub cache** to refresh when the remote **Hub revision** changes, so that a bumped `stable` tag is picked up without stale weights.
6. As a researcher running transcribe, I want segment and transcribe jobs to succeed when the **Registry** points at Hub, so that I get the same API behavior as with local weights.
7. As a developer without Hub credentials, I want **local bundled weights** under the Hub layout to satisfy `file://` entries, so that Docker Compose and local pytest work offline.
8. As a security-conscious operator, I want only allowed URI schemes (`file://`, `package://`, `hf://`) to resolve to loadable paths, so that arbitrary remote URLs cannot be injected via the **Registry**.

### Publishing and discovery

9. As a model trainer, I want training output copied into the **Hub staging tree**, so that publish-ready **Hub artifacts** have a consistent layout before upload.
10. As a model trainer, I want a publish script that pushes one **registry model id** + **registry tag** from staging to the correct **Hub model repo**, so that releasing a model is repeatable.
11. As a model trainer, I want each **Hub model repo** to include a model card describing script, architecture, and **registry model id**, so that Hub visitors understand what the checkpoint is for.
12. As a corpus curator, I want labelled line crops publishable to a **Hub dataset repo** with a searchable **Hub dataset slug**, so that training data is discoverable separately from weights.
13. As a project maintainer, I want a **Hub collection** (`nomos`) listing our **Hub model repos** and **Hub dataset repos**, so that users can browse the ecosystem on Hugging Face.
14. As a project maintainer, I want collection metadata versioned in-repo and syncable via script, so that collection changes are reviewed in git before pushing to Hub.

### Registry and platform alignment

15. As a platform admin, I want **registry model ids** to follow `{script}-{architecture}-{model_version}`, so that naming is consistent between **Registry**, **Hub repo slug**, and cache paths.
16. As a platform admin, I want legacy ids (e.g. `greek-calamariv1`) migrated or aliased, so that existing **InferenceModel** seeds and docs do not break silently.
17. As a platform admin, I want dev seed data to reference the same **registry model ids** as `registry.yaml`, so that platform jobs and inference agree on which model ran.
18. As an operator reading logs, I want weight resolution failures (missing Hub repo, auth, corrupt cache) to surface clear errors on the inference job, so that I can fix configuration without digging into stack traces.

### Operations and environments

19. As a CI maintainer, I want tests to mock Hub downloads at the integration boundary, so that CI does not require Hub tokens or network for every run.
20. As a CI maintainer, I want at least one integration test that exercises real `hf://` resolution against a small public or fixture repo, so that regressions in download/cache logic are caught.
21. As a Docker operator, I want Compose to mount or populate **local bundled weights** for default models, so that `docker compose up` works without `HF_TOKEN`.
22. As a GPU operator, I want cached weights on a persistent volume keyed by **registry model id** and **registry tag**, so that multi-worker inference shares one download.

### Documentation and vocabulary

23. As a new contributor, I want `inference/CONTEXT.md` vocabulary to match the implemented layout, so that **Hub integration**, **Hub cache**, and **Hub staging tree** refer to real paths.
24. As a new contributor, I want a short runbook for first-time Hub publish (namespace, token, staging → push), so that releasing a model does not depend on tribal knowledge.

## Implementation Decisions

- **Module boundary**: **Hub integration** lives under `src/hf/` (resolver, cache, manifest, Hub client wrapper). Inference continues to call a single weight-resolution entry point; `hf://` handling is delegated there rather than embedded in architecture adapters.
- **URI contract**: `hf://<namespace>/<hub-repo-slug>@<registry-tag>` maps to **Hub model repo** `{namespace}/{hub-repo-slug}` at git tag/revision equal to **registry tag** (e.g. `stable`). Parsing and validation follow `inference/CONTEXT.md`.
- **Cache layout**: **Hub cache** at `src/hf/cache/<registry_model_id>/<registry_tag>/` with **Hub cache manifest** (e.g. `.hub-manifest.json`) recording revision id and content hash. Reuse requires manifest match, not merely directory existence.
- **Local bundled weights**: Offline dev and Docker defaults use `src/hf/local/{script}/{architecture}/{model_version}/{registry_tag}/` with `file://` URIs relative to `src/hf/`. Existing `inference/weights/` content migrates incrementally; `package://kraken/...` remains supported for Kraken until a Hub-backed segment model ships.
- **Lazy vs prefetch**: Inference job runner resolves weights at job execution time (lazy fetch). `scripts/hf/fetch_model.py` provides explicit prefetch for operators.
- **Hub client**: Use `huggingface_hub` for download/list revision metadata; wrap behind a small interface so unit tests mock one seam.
- **Registry migration**: Introduce new **registry model ids** (`greek-calamari-v1`, `syriac-calamari-v1`, `greek-kraken-segment-v1` or equivalent per CONTEXT) in a late slice; keep backward-compatible aliases or a single release cutover documented in the migration issue.
- **Platform catalog**: No change to HTTP contracts between platform and inference. Update dev seeds and docs so **InferenceModel** `artifact_ref` values match post-migration **registry model ids**.
- **Secrets**: `HF_TOKEN` (or Hub CLI login) required only for private repos and publish scripts; public model repos should work for read without token.
- **Staging → Hub publish**: Scripts under `scripts/hf/` (`publish_model.py`, `publish_dataset.py`, `sync_collection.py`). **Hub staging tree** under `src/hf/staging/models/` and `src/hf/staging/datasets/`.
- **Collection**: Source of truth `src/hf/collection.yaml`; collection slug `nomos` on the configured **Hub namespace**.
- **Calamari artifact shape**: Publish TensorFlow SavedModel directories plus JSON sidecar per CONTEXT (**Hub artifact**), not forced `model.ckpt` rename.
- **Out of scope for v1 Hub integration**: Training pipelines writing directly to Hub; automatic platform UI for model version picker; TrOCR or non-Kraken/Calamari architectures; moving Calamari off TensorFlow.

## Testing Decisions

- **Primary seam — weight resolution**: Extend `tests/inference/unit/test_registry.py` (and new `tests/hf/` or `tests/inference/unit/test_hf_weights.py`) to assert `hf://` parsing, cache hit/miss, manifest invalidation, and that `file://` / `package://` behavior is unchanged. Mock `huggingface_hub` at the download/metadata boundary — do not test Hub SDK internals.
- **Secondary seam — job runner**: One integration test in `tests/inference/integration/` that runs transcribe (or segment) with a **Registry** entry pointing at `hf://` where the Hub client is mocked to materialize files into **Hub cache**; assert the architecture receives a real filesystem path.
- **E2E optional lane**: One marked test (skipped in default CI without env flag) that fetches a tiny public Hub repo or recorded fixture archive, to validate real network + manifest flow.
- **Publish scripts**: Test staging layout validation and dry-run mode without calling Hub upload API in default CI; upload tests mock Hub API or run only in manual/scheduled workflow.
- **Good test bar**: Tests assert observable behavior (resolved path exists, cache reused on second resolve, error message on bad URI), not internal call order of the Hub client.

## Out of Scope

- Hugging Face Spaces or Gradio demos for inference.
- Hosting the annotation platform on Hub.
- Automatic sync from **Export** artefacts to **Hub dataset repos** (manual/scripted publish only in v1).
- Replacing the platform Postgres **InferenceModel** catalog with Hub as source of truth.
- ARM-optimized Calamari rewrite (separate future work).
- UI for browsing Hub models in the annote editor.

## Further Notes

- Domain vocabulary is authoritative in `inference/CONTEXT.md`; this PRD operationalizes that document.
- `docs/todo.md` already calls out remote weight flow and Hub as a model repository — this PRD supersedes that bullet for planning purposes.
- **Hub namespace** may start under a personal account and later move to `nomicous` org without changing **Hub repo slug** conventions.
- Prefer thin vertical slices (tracer bullets): first slice proves `hf://` → cache → inference load for one model; later slices add publish, datasets, collection, and registry id migration.
