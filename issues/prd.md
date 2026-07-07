# greekOCR Platform — Product Requirements Document

## Problem Statement

Byzantine Greek manuscript researchers ( and other researchers as well ) need to transcribe difficult handwriting ( of manuscripts ) at scale. Today they rely on ad hoc scripts, brittle one-off UIs, and models that are poorly matched to Greek (high CER/WER). Segmentation (e.g. Kraken) can work well, but recognition quality, data organization, model comparison, and expert-in-the-loop correction lack a single durable system.

Researchers cannot easily: organize work by project; keep page-level layout (blocks and lines) consistent with scholarly structure; run and compare multiple OCR/HTR models; preserve human corrections when re-running models; share finished work publicly without giving strangers edit access; or build a growing corpus of curated labels for future training.

## Solution

Build **greekOCR** as a hosted **Platform** (not just a pipeline script) that mirrors the proven eScriptorium document hierarchy—**Project → Document → DocumentPart → Block → Line**—with named **Transcription** layers, Postgres-backed **Jobs** for long ML work, and a Next.js annotation UI ported conceptually from eScriptorium (geometry JSON compatible, React not Vue).

The platform separates **Layout edit** (manual geometry on blocks/lines) from **Transcription edit** (human text in a single **Ground truth** layer per document). Model runs never overwrite human layout; **Segment merge** updates, adds, or prunes only machine-generated geometry. Each **Transcribe** job creates a new model layer; humans curate via **Copy to ground truth** and direct edits. **Published** documents are world-readable and read-only for non-project users; project members retain full edit rights.

## User Stories

### Accounts and access

1. As a researcher, I want to register and sign in with JWT, so that my projects and documents are private by default.
2. As a project owner, I want to create a **Project** workspace, so that I can group related manuscripts.
3. As a project owner, I want to share a project with other users by username, so that collaborators can edit without team/group complexity in v1.
4. As a collaborator, I want to see only projects shared with me or owned by me, so that I do not access others’ draft work.
5. As a project member, I want to read and edit all documents in the project, so that collaboration matches eScriptorium-style shared workspaces.
6. As a visitor without project access, I want to read a **published** document in **Public view**, so that I can cite or study shared transcriptions without an account.
7. As a visitor, I must not edit layout, text, or run ML jobs on **published** documents, so that public corpora stay trustworthy.
8. As a project member, I want to keep editing a **published** document, so that publication does not freeze ongoing curation for the team.
9. As a document owner, I want to set workflow to `draft`, `published`, or `archived`, so that I control visibility and catalog presence.
10. As a catalog browser, I want **archived** documents hidden from normal lists, so that retired work does not clutter discovery.

### Projects and documents

11. As a researcher, I want to create a **Document** under a project, so that one codex or manuscript is one logical unit.
12. As a researcher, I want to upload or attach images as ordered **DocumentParts**, so that folio order is preserved.
13. As a researcher, I want to add guidelines or notes at project level, so that collaborators follow the same transcription rules.
14. As a researcher, I want to open a document dashboard listing parts and processing status, so that I know which pages need work.
15. As a researcher, I want to delete or reorder parts when I uploaded wrong scans, so that the catalog stays accurate.

### Layout and segmentation

16. As a researcher, I want to run **Segment** on a page (DocumentPart), so that blocks and lines are detected automatically.
17. As a researcher, I want segment results to include both **Blocks** and **Lines** (full hierarchy), so that layout matches scholarly regions not only baselines.
18. As a researcher, I want to switch the UI to **Layout edit** mode, so that I can correct baselines, masks, and block boxes on the canvas.
19. As a researcher, I want my layout saves to set **Manual geometry**, so that the system knows my edits are authoritative.
20. As a researcher, I want to re-run **Segment** after manual edits without losing my adjusted lines, so that I can try better models safely.
21. As a researcher, I want re-segment to update machine lines, add new machine lines, and prune obsolete machine lines, so that the layout stays current without touching manual work.
22. As a researcher, I want pruned machine lines to remove their transcriptions in all layers, so that the database stays consistent.
23. As a researcher, I want **Reset layout** on selected lines or a whole page, so that I can deliberately allow the next segment job to replace geometry.
24. As a researcher, I want segment jobs to return immediately with a job id, so that I am not blocked on long Kraken runs.
25. As a researcher, I want to poll job status until segment completes, so that I know when to refresh the canvas.

### Transcription and models

26. As a researcher, I want to run **Transcribe** on a part or document, so that line text is generated from layout.
27. As a researcher, I want each transcribe run to create a new named **Transcription** layer, so that I can compare model outputs over time.
28. As a researcher, I want **Transcribe** never to write into **Ground truth**, so that human curation stays separate from model output.
29. As a researcher, I want exactly one **Ground truth** layer per document, so that there is a single canonical human transcription.
30. As a researcher, I want **Ground truth** to start empty, so that I am not surprised by auto-filled text.
31. As a researcher, I want **Copy to ground truth** from a model layer (whole doc or selection), so that I can bootstrap curation from the best run.
32. As a researcher, I want to switch to **Transcription edit** mode and type corrections only in **Ground truth**, so that layout and text editing are not confused.
33. As a researcher, I want to view multiple transcription layers side by side or via layer picker, so that I can compare Kraken, TrOCR, and future models.
34. As a researcher, I want per-line confidence stored on model **LineTranscription** rows, so that I can prioritize review of low-confidence lines.
35. As a platform admin (v1: developer), I want an **InferenceModel** catalog entry for each runnable model, so that providers (Kraken, TrOCR, Hugging Face) are registered consistently.
36. As a researcher, I want **ModelBinding** at project, document, or part scope, so that defaults can be set once and overridden per page.
37. As a researcher, I want the most specific binding to win, so that one difficult folio can use a different segment model.
38. As a researcher, I want jobs to record which model/binding ran, so that results are reproducible.

### Jobs and pipeline

39. As a researcher, I want binarize → segment → transcribe as a coherent **Pipeline**, so that preprocessing matches training workflows.
40. As a researcher, I want pipeline steps invoked via API and **Job** records, so that the same orchestration works for UI and scripts.
41. As an operator, I want the API to run multiple Uvicorn workers with jobs claimed from Postgres, so that one worker crash does not lose the job queue.
42. As a researcher, I want failed jobs to surface error messages, so that I can fix model paths or GPU issues.
43. As a developer, I want **ModelAdapter** modules to map provider output to **canonical segment/transcribe DTOs**, so that new engines integrate without changing persistence rules.

### Comparison, export, and corpus building

44. As a researcher, I want to compare metrics (CER/WER) between layers or against **Ground truth**, so that I can measure model improvements (when metrics tooling exists).
45. As a researcher, I want to export a document’s layout and transcriptions in a standard format, so that I can publish datasets and train models offline.
46. As a researcher, I want curated **Ground truth** to feed training pipelines under `ocr/`, so that annotation directly improves Greek HTR models.
47. As a researcher, I want to experiment with Google OCR APIs and open VLMs later via new adapters, so that the platform is not locked to Kraken/TrOCR.

### UI and developer experience

48. As a frontend developer, I want OpenAPI-generated types from the FastAPI schema, so that the Next.js client stays aligned with the API.
49. As a frontend developer, I want to port eScriptorium canvas/geometry patterns to React, so that proven annotation UX is reused without Vue.
50. As a developer, I want bounded contexts (users, project, document, inference), so that the backend stays maintainable under DDD.
51. As a developer, I want Alembic migrations on PostgreSQL, so that schema evolution is explicit and reviewable.
52. As a developer, I want Docker Compose for Postgres and API locally, so that onboarding matches production-shaped structure.

### Legacy prototype migration

53. As a developer, I want the old upload/segment/transcribe prototype endpoints deprecated in favor of project/document APIs, so that one platform API exists.
54. As a researcher who used the Vite prototype, I want equivalent or better canvas editing after migration, so that workflow is not a regression.

## Implementation Decisions

### Architectural shape

- **Bounded contexts**: `users` (auth, identity), `project` (workspace, sharing), `document` (hierarchy, workflow, transcriptions, media), `inference` (model catalog, bindings, jobs, adapters).
- **Layers per context**: `domain` (entities, rules), `application` (use cases, merge policies), `infrastructure` (context ORM, storage, adapters), `api` (FastAPI routers, Pydantic request/response DTOs).
- **Shared `infrastructure/` (repo root)**: Postgres engine (`db.py`), settings (`config.py`), Alembic metadata aggregator (`models.py`), **`infrastructure/alembic/`** for migrations — not under `backend/` or any bounded context.
- **Stack**: FastAPI + Pydantic v2, SQLAlchemy 2 async + sync URL for Alembic, PostgreSQL, Uvicorn with four workers, Next.js App Router frontend, Docker Compose (Postgres + API in v1). No Celery/Redis in v1.
- **Reference**: eScriptorium domain model and UI semantics; implementation is new (FastAPI/Next), not Django/Vue embedding.

### Deep modules (encapsulated, testable interfaces)

| Module | Responsibility | Interface sketch |
|--------|----------------|------------------|
| **Access policy** | Decide read/write/public for project, document, job | `can_read(user, resource)`, `can_write(user, resource)` |
| **Job runner** | Enqueue, claim (`FOR UPDATE SKIP LOCKED` or equivalent), transition status, invoke handler | `enqueue(type, payload)`, `claim_next()`, `complete(id, result)` |
| **Segment merge** | Apply canonical segment DTO to existing part: respect manual flag, update/add/prune machine entities, cascade-delete transcriptions on pruned lines | `apply(part_id, canonical_segment, job_id)` |
| **Transcription layer factory** | Create new layer per transcribe job; ensure ground truth singleton per document | `create_model_layer(document_id, job_meta)`, `get_or_create_ground_truth(document_id)` |
| **Copy to ground truth** | Copy line texts from source layer to ground truth with overwrite rules | `copy(document_id, source_layer_id, line_ids?)` |
| **Model binding resolver** | Resolve segment/transcribe model for part/document/project | `resolve(task, part_id) -> model_id, config` |
| **Model adapter registry** | Map provider + task to adapter; validate canonical DTOs | `run(adapter_key, input) -> CanonicalSegmentResult \| CanonicalTranscribeResult` |
| **Media store** | Store and serve DocumentPart image bytes | `put(part_id, stream)`, `url(part_id)` |
| **Auth service** | JWT issue/validate, password hash | `login`, `current_user` dependency |

Pipeline code under `ocr/` implements adapters only; it does not import FastAPI.

### Domain persistence (conceptual schema)

- **User**: id, username, email, hashed password.
- **Project**: owner, shared users (M2M), slug, guidelines.
- **Document**: project_id, name, workflow (`draft` \| `published` \| `archived`).
- **DocumentPart**: document_id, order, image storage key, dimensions optional.
- **Block**: part_id, box polygon JSON, manual_geometry flag.
- **Line**: part_id, block_id optional, baseline/mask JSON, manual_geometry flag.
- **Transcription**: document_id, name, kind (`ground_truth` \| `model`), created_by_job_id optional.
- **LineTranscription**: line_id, transcription_id, text, confidence.
- **InferenceModel**: provider, task, artifact reference, default params JSON.
- **ModelBinding**: scope (project/document/part), task, model_id, overrides JSON.
- **Job**: type, status, payload JSON, result JSON, error, model/binding reference, timestamps, optional user/document/part FKs.

### API contracts (v1 surface, behavioral)

- **Auth**: register, login, `GET /me`.
- **Projects**: CRUD, share/unshare user by username.
- **Documents**: CRUD under project, workflow transition, list parts.
- **Parts**: upload image, reorder, delete.
- **Layout**: CRUD blocks/lines (layout edit saves set manual flag); reset layout action.
- **Transcriptions**: list layers; patch ground truth line texts; copy-to-ground-truth action.
- **Inference**: list models; CRUD bindings; `POST .../segment`, `POST .../transcribe` → `{ job_id }`; `GET /jobs/{id}`.
- **Public**: `GET` published document + parts + read-only geometry and transcription layers (no mutating routes without auth + membership).

OpenAPI export drives frontend type codegen.

### Segment and transcribe behavior (locked)

- **Segment merge**: manual geometry immutable; machine geometry merge with prune; prune cascades all **LineTranscription** for that line.
- **Transcribe**: always new model **Transcription** layer per job.
- **Ground truth**: single per document; UI transcription edit writes only here; populated by explicit copy or typing.
- **Published**: **Public view** read-only for non-members; members retain write/job rights.

### Job execution

- Jobs persisted in Postgres; handlers run in API process pool across Uvicorn workers.
- Claim must be transactional so two workers do not run the same job.
- HTTP handlers enqueue and return `job_id`; clients poll until `done` or `failed` (**current**).
- **Target:** Postgres `NOTIFY` on `jobs` status changes + SSE to the browser — see [`docs/decisions/001-platform-job-status-push.md`](../docs/decisions/001-platform-job-status-push.md).
- v1 does not block HTTP until Kraken/TrOCR finishes for large batches.

### Frontend (structural, not pixel-perfect in v1)

- Next.js App Router; routes for project list, project dashboard, document editor.
- Editor: two modes — **Layout edit** and **Transcription edit** (toolbar toggle).
- Canvas: port eScriptorium geometry editing concepts (polygons/baselines) to React; reuse JSON shapes for interoperability.
- Job panel: poll active jobs; toast on failure.
- Layer picker: model layers read-only in transcription edit except ground truth; model layers editable only via copy source, not direct typing (unless explicitly viewing for compare — compare is read-only view).

### Migration from prototype

- Existing FastAPI upload/segment/transcribe on ephemeral `image_id` remains until document/part APIs subsume it.
- Kraken adapter must emit full block+line **canonical segment result** (prototype line-only extraction is not the target behavior).

## Testing Decisions

### What makes a good test

- Test **observable behavior** through application service and HTTP boundaries: status codes, persisted state, access denials, job lifecycle.
- Do not assert internal ORM call order or private helper names.
- Prefer one failing example per rule plus several passing cases (per project TDD note); expand when modules stabilize.

### Modules to test first (recommended)

| Module | Priority | Example cases |
|--------|----------|---------------|
| **Segment merge** | High | Manual line untouched when re-segment; machine line pruned deletes transcriptions; new machine line added |
| **Access policy** | High | Non-member read published; non-member cannot POST segment; member can edit published |
| **Job runner** | High | Double claim does not duplicate work; failed job stores error |
| **Model binding resolver** | Medium | Part overrides document overrides project |
| **Copy to ground truth** | Medium | Copies selected lines; does not alter layout |
| **Transcription layer factory** | Medium | Second transcribe job creates second model layer; only one ground truth |

### Prior art

- Prototype backend has little automated test coverage today; new tests live alongside bounded context `application` packages.
- `ocr/` pipeline scripts may have ad hoc evaluation; platform tests focus on merge, access, and job contracts—not GPU inference accuracy.

### Explicitly not required in v1

- End-to-end browser tests (optional later).
- ML accuracy regression suites in CI (research workflows stay offline until stabilized).

## Out of Scope

- **Teams**, group sharing, email invitations, quotas.
- **Celery**, Redis job broker, separate worker container (deferred; structure allows adding later).
- **Monitoring** (Prometheus, Sentry).
- **OpenSearch/Elasticsearch** document search.
- **Train** jobs in UI (adapter hook reserved; training orchestration later).
- **IIIF** import (optional future; v1 is file upload).
- **Public view** nuances: which layers appear by default, export formats, citation widgets — refine in production.
- **Fork published document to my project** — deferred.
- **Orphan lines** without blocks except explicit edge cases — default requires block linkage once layout established.
- **Auto-seed ground truth** on first open.
- **Inferring manual geometry** from job diff (explicit flag only in v1).

## Further Notes

- **Production nuances** (public layer visibility, export formats, rate limits, GPU node placement) are intentionally deferred; this PRD locks **structure and invariants** so implementation does not bake in wrong defaults.
- **CONTEXT.md** is the glossary source of truth for terms; keep it free of file-level implementation detail. Update it when domain language changes.
- **Comparison tooling** (CER/WER between layers) is a high-value researcher feature aligned with README goals; schedule after core platform CRUD + jobs + editor modes exist.
- **Greek-specific models** (fine-tuned TrOCR, XLM-R, Calamari, VLMs) integrate via **InferenceModel** + **ModelAdapter** without changing **Segment merge** or **Ground truth** rules.
- Consider an ADR when introducing a separate worker service or Redis queue — hard to reverse once multi-node deployment depends on it.
