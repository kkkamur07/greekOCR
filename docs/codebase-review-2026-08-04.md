# Codebase review, 2026-08-04

A snapshot review covering repository cleanup, code simplification, and architectural
depth. Produced by nine parallel audits (structure, git hygiene, lint and test health,
documentation prose, source comments, and four simplification passes), followed by three
architectural depth passes framed against `nomicous/CONTEXT.md` and `inference/CONTEXT.md`.

Line numbers are accurate as of commit `21c24b2` with the working tree as it stood on
2026-08-04. Neither the line numbers nor the file paths survived: roughly ten findings
below cite files that have since been deleted, among them `inference/helper/routes/info.py`,
`inference/api/jobs.py`, `nomicous/frontend/src/inference/helperInfo.ts`,
`packaging/helper/excludes.txt` and `tests/inference/unit/test_helper_app.py`. The header
used to call file paths durable; that was true when it was written and is not true now.

**This document is stale.** It predates ADR 0004 and the `src/` constraint. Treat it as a
record of what was believed on 2026-08-04, not as a work list. For the current state of
the redesign read [`resume-inference-redesign.md`](./resume-inference-redesign.md).

Claims are marked **[verified]** where they were re-checked directly against the working
tree, and **[reported]** where they come from an audit pass and have not been independently
confirmed. Two agent claims were found wrong on verification and are recorded in
[Corrections](#corrections) so nobody re-derives them.

## Contents

- [Do these first](#do-these-first)
- [The diagnosis](#the-diagnosis)
- [Deepening candidates](#deepening-candidates)
- [Decisions that are yours, not the code's](#decisions-that-are-yours-not-the-codes)
- [Leave these alone](#leave-these-alone)
- [Cleanup backlog](#cleanup-backlog)
- [Corrections](#corrections)

## Do these first

### The working tree cannot be reconstructed from git

**[verified]** 44 untracked files are imported by tracked, modified files. Committing only
the tracked changes produces a repository that fails at import.

- `nomicous/backend/core/app.py:35` imports `backend.core.settings.device` (untracked)
- `app.py:47-49` import `backend.ml.api.device_pairing`, `device_self`, `devices` (untracked)
- `jobs/api/jobs.py` and `jobs/application/job_service.py` import
  `backend.jobs.infrastructure.stale_sweep` (untracked)
- Alembic migrations `003`, `004`, `005` are untracked while their ORM changes are tracked,
  so the schema chain is incomplete

**[verified]** Local `main` is 2 commits behind `origin/main` and 0 ahead, with roughly 117
files dirty. One incoming commit removes the legacy `configs/` directory, which is still
tracked locally, so the pull will interact with the working tree.

A fix that unblocks the entire Kraken segmentation pipeline currently exists only on local
disk: `src/model/kraken/finetuning.py` imports `model.transcription.*` in `HEAD`, a path that
has never existed, so `fine_tune()` has raised `ModuleNotFoundError` since commit `168adb8`.
The working tree corrects it to `src.model.kraken.*`. Until that is committed,
`configs/kraken_seg.yaml` (67 lines of tuned hyperparameters) remains unreachable.

Every candidate in this document assumes a committed baseline.

### The blla-segment registry pin has drifted

**[verified]** `inference/registry.yaml` pins `blla-segment` to an artifact digest that
matches neither staged file.

| Source | SHA-256 (first 8) |
|---|---|
| `registry.yaml` `artifact_sha256` | `5871e375` |
| `src/hf/staging/models/segmentation/blla/v1/stable/blla.onnx` | `d3e9c086` |
| `src/hf/staging/models/segmentation/blla/v1/stable/blla.safetensors` | `8b5b6ec2` |

The mechanism works when a human gets it right: `syriac-calamari-v1`'s pin `3cb01b58…`
matches its staged `best.onnx` byte for byte. But nothing detects the failure case. No test,
script, or CI check compares a registry pin to a staging artifact.

This needs a decision: is the published Hub artifact correct and staging stale, or the
reverse? See [candidate 4](#4-hub-integration-one-call-and-a-type-that-says-verified) for
the durable fix.

## The diagnosis

One sentence covers most of what the audits found: **the same decision is re-made at two or
three different depths, and no module owns it.**

Which model, on which host, is this payload admissible, is this job still alive, is this
Segment paired, what number is this Segment. Each is decided repeatedly, in different layers,
sometimes in different languages.

**[reported]** One "segment this page" click crosses roughly 30 files and 5 processes, with
3 separate image validations, 2 registry resolutions, and 3 transactions on the callback
side. The trace is not long because the problem is hard.

The baseline is strong, which is what makes the architectural work worth doing rather than
the cleanup work:

- Zero `any`, zero `@ts-ignore`, zero commented-out code, zero TODO/FIXME markers in
  first-party source
- 583 tests collect with 0 errors; zero unconditional skips, zero `xfail`, zero `assert True`
- `tsc --noEmit` clean; eslint reports 3 warnings
- Comment density 1.5% in `nomicous/backend`, 0.9% in `inference`, and the comments that
  exist are the useful "why" kind

Seven of eleven AI-writing tell categories return zero across the source tree. There is no
slop cleanup to do in code comments. See [Documentation](#documentation) for where the prose
problems actually are.

## Deepening candidates

Ranked by friction removed per unit of risk. Vocabulary follows `CONTEXT.md` for the domain
and the deepening glossary for architecture: a **module** has an interface and an
implementation, an interface is everything a caller must know, **depth** is a lot of
behaviour behind a small interface, a **seam** is where an interface lives.

| # | Candidate | Area | Risk | Ratio |
|---|---|---|---|---|
| 1 | Segment number becomes a module | frontend | very low | highest |
| 2 | Make `inference/contracts/` actually portable | seam | low-med | very high |
| 3 | One lease primitive, two adapters | jobs | medium | high |
| 4 | Hub integration: one call, a verified-artifact type | hub | medium | high |
| 5 | Give Pairing a module | frontend | medium | high |
| 6 | Extract shared Product job run orchestration | frontend | low-med | high |
| 7 | Give the Product job lifecycle a home | jobs | med-high | medium |
| 8 | Give the Registry a lifetime | hub | medium | medium |
| 9 | Decide Host eligibility in Python | seam | low | medium |
| 10 | Type the platform API seam from the generated schema | frontend | low-med | medium |

### 1. Segment number becomes a module

**Files:** `PageEditorCanvas.tsx:295`, `usePairingState.ts:121-131`,
`useLayoutMutations.ts:161,283`, `PageEditorSegmentNav.tsx`.

**Problem.** Three mutually inconsistent formulas compute one glossary concept. The canvas
uses `line.order + 1`, the strip uses index-in-sorted-array + 1, and a new Segment takes
`order = sortedLines.length`.

**[verified]** `layout_service.py` `delete_part_line` deletes and commits without
renumbering. So after deleting one of four Segments, orders are `0,1,3`: the canvas announces
segments 1, 2, 4 while the strip announces 1, 2, 3, and a newly drawn Segment is assigned
`order: 3`, colliding with the existing one. Order-sort then becomes unstable, so segment
navigation can address the wrong Segment.

Segment number is the export filename (`<manuscript_name>_<segment_number>`). This is data
correctness, not presentation.

**Solution.** One place answers "what number is this Segment on this Page, and what number
does the next one get". Renumbering on delete belongs to the backend; the frontend should
stop inferring the next order from array length.

**Benefits.** *Locality:* one definition instead of three. *Leverage:* the reading-order
question already flagged in `CONTEXT.md:32` becomes one edit. *Testability:* a table test
over gapped orders replaces DOM tests that cannot currently observe the disagreement.

### 2. Make `inference/contracts/` actually portable

**Files:** `inference/contracts/common.py:13-14,51`, `contracts/jobs.py:11,15`,
`deploy/platform/build.sh:54-76`, `nomicous/Dockerfile:42`, `.dockerignore`,
`tests/nomicous/unit/test_platform_bundle.py`.

**Problem.** `contracts/` is named as the platform-to-inference seam but imports
`inference.admission` and `inference.infrastructure.settings.get_inference_settings` at
module scope, and calls the latter inside a pydantic validator. The platform process
instantiates an inference settings object holding a database URL, a webhook secret, and pool
sizes it never uses, merely by parsing a response.

**[reported]** `build.sh:54-76` is not a curated list. It is the transitive import closure of
`inference.contracts`, which is why "what the platform needs from inference" is defined in
four places that can drift: `build.sh`, `nomicous/Dockerfile:42` (`COPY inference`, meaning
everything), `.dockerignore`, and `test_platform_bundle.py`.

Only `segment.py`, `transcribe.py`, and `webhooks.py` are genuinely portable today.

**Solution.** Let contract types describe shape only. Make admission an explicit step the
caller invokes with limits it passes in, rather than something a validator reaches out for.

**Benefits.** *Locality:* one definition of the dependency instead of four. *Leverage:* the
contract becomes shippable, so the four bundle definitions collapse to one line.
*Testability:* contract round-trip tests stop needing environment variables, and admission
limits become a parameter rather than a cache to clear.

### 3. One lease primitive, two adapters

**Files:** `inference/infrastructure/job_repository.py:68-121`,
`nomicous/backend/jobs/infrastructure/job_repository.py:137-199`.

**Problem.** `claim_next_pending_job`, `reclaim_stale_running_jobs`, and
`seconds_until_next_stale_running_job` exist twice with identical SQL predicates, against the
same Postgres, differing only in table name and whether `claimed_by`/`heartbeat_at` are set.

**[verified]** The two services share one database. `docker-compose.yml:113-114` sets
`DATABASE_URL` and `INFERENCE_DATABASE_URL` to byte-identical connection strings, and
`_create_inference_jobs()` at `001_initial_schema.py:663` creates the inference service's
table inside the platform's own migration. The same file at line 468 carries the comment
"inference_jobs lives in the inference service's own schema and may be a different database
entirely", which the file then contradicts 200 lines later.

A change to the claim ordering or the lease predicate has to land in two files, enforced only
by comment discipline.

**Solution.** Extract the lease behaviour, meaning claim one row with `SKIP LOCKED` in FIFO
order, expire a lease, report the next deadline, as one module parameterised by table and
status column. Two adapters already exist, so this is extraction rather than speculation.

The lifecycles above the lease are genuinely different concepts. A Product job carries user
ownership, cancellation, waiting-on-a-delegate, SSE fan-out, and a document merge. An
Inference job carries image bytes, a registry selection, and one outbound webhook. Do not
merge those.

**Benefits.** *Locality:* `SKIP LOCKED` semantics in one place. *Leverage:* both workers get
identical fencing; a third queue costs one adapter. *Testability:* lease expiry and claim
contention become testable once.

### 4. Hub integration: one call, and a type that says verified

**Files:** `src/hf/resolve/`, `inference/jobs/runner.py:154-191`,
`inference/helper/routes/info.py:66-91`, `scripts/hf/fetch_model.py:38-62`.

**Problem.** "Give me a verified local artifact path for this Registry model id and Registry
tag" is not a call. It is a seven-step remembered ritual, and the three callers remember it
differently.

**[reported]** `entry.versions[tag]` is indexed twice because `get_model_entry` validates the
tag then discards the result and returns the model. Six arguments are re-marshalled out of an
object the caller already holds, with the architecture enum flattened to a bare string. The
return is a plain `Path`, so the fact that the artifact was verified is lost and the digest
gets checked two more times downstream, which is why the memoization patch at
`artifacts.py:26-65` exists. Callers then branch on `weights_path.suffix == ".onnx"` to
re-derive the artifact format that `find_hub_artifact` already chose.

`info.py:66-91` skips the middle steps entirely and hand-rebuilds the cache check, including
hand-parsing the `hf://` URI at line 78 rather than calling the exported
`parse_hf_weights_uri`.

The publish side never closes the loop. `resolve_hf_weights_source` hard-requires
`hub_revision` and `artifact_sha256`, and no tool in the repository produces them.
`src/hf/publish/model.py:81` receives the commit sha from `upload_folder`, uses it for
`create_tag`, and discards it. The documented procedure is to run a one-liner and copy two
values by hand. The `blla-segment` drift recorded above is the consequence.

**[reported]** `find_hub_artifact` has ruff complexity 26 against a limit of 10 and zero
direct tests. Its `architecture=None` default selects an mlmodel-only lookup that can never
return an ONNX artifact, and both `resolve_hf_weights_source` and `resolve_weights_source`
default that argument.

**Solution.** One entry point taking a Registry model id and Registry tag, returning a value
carrying the artifact path, its verified digest, the architecture, and the format, so no
caller re-verifies, re-indexes, or re-derives. Separately, make publish emit the exact
registry stanza, and add a test asserting every `hf://` entry's digest matches its staging
artifact.

Characterize `find_hub_artifact` with tests before changing it. The alias divergence may be a
deliberate Kraken-era migration artifact.

**Benefits.** *Locality:* ordering moves from three developers' heads into one function.
*Leverage:* the largest single friction reduction in the inference tree. *Testability:* one
seam to mock instead of five primitives, and the cross-store invariant becomes a unit test.

### 5. Give Pairing a module

**Files:** `usePairingState.ts`, `hooks/utils.ts`,
`PageEditorTranscriptionStrip.tsx:71-119`, `PageEditorToolbar.tsx:150-155,179-191`,
`PageEditorPlaceholderPage.tsx:282,311-321`, `PageEditorCanvas.tsx:272-332`,
`usePageEditorData.ts`, `useLayoutMutations.ts`.

**Problem.** Pairing, the central concept in `nomicous/CONTEXT.md`, has no seam. Eight
modules must change to alter one Pairing rule, and they disagree about who owns truth: two
compute it locally, two read it from the server, one invalidates it.

**[reported]** Pairing progress is implemented three times: authoritative SQL in
`document_service_shared.py:136-148`, a TypeScript restatement of the same predicate in
`utils.ts:23-30`, and a recomputation in `PageEditorToolbar.tsx:150-155` that discards the
`percent` the backend already sent in the same payload. The canvas colours from optimistic
local state while the counter comes from a later round-trip, so they can visibly disagree.

The accept-mode state machine deciding when a Model transcription becomes a Ground truth
transcription, the rule `CONTEXT.md:306` singles out as historically ambiguous, lives inside
a render function.

**Solution.** One module owns which Segment is selected, its draft text, which Transcription
layer is in view, whether the Segment counts as paired, the Page's Pairing progress, and what
the accept action does. Canvas, strip, and toolbar consume the answer rather than the raw
materials.

**Benefits.** *Locality:* one file per Pairing rule change instead of eight. *Leverage:* the
toolbar's 8 pairing props, the strip's 14, and the canvas's 3 collapse; the canvas-versus-
toolbar disagreement becomes impossible by construction. *Testability:* the accept-mode
machine becomes a table test, and 654 lines of DOM-driving transcription tests shrink to a
few integration checks.

### 6. Extract shared Product job run orchestration

**Files:** `usePairingState.ts:339-482`, `useLayoutMutations.ts:507-646`.

**Problem.** Roughly 110 lines duplicated between the transcribe path and the segment path,
covering the hardest reasoning in the frontend: three-way abort discrimination (user cancel,
cloud switch, superseded), the persisted-versus-not gate that prevents paying twice, and the
local-only routing error. Both copies carry the same comments verbatim.

**[reported]** The two test files test the same seven scenarios twice, once per copy. A fix
to one is a silent bug in the other.

**Solution.** One module owns "run this task locally, and if it did not persist, hand it to a
cloud Product job". Segment and transcribe describe what to run and what to do with the
result.

**Benefits.** *Locality:* one place knows why a run stopped. *Leverage:* the third task
(binarize, already present in `inference/types.ts:1`) gets it free. *Testability:* seven
duplicated scenarios collapse to seven, tested once, with no React.

### 7. Give the Product job lifecycle a home

**Files:** `nomicous/backend/jobs/infrastructure/job_repository.py:202-262,293-378`,
`jobs/application/job_callback_service.py:234-357`, `jobs/infrastructure/stale_sweep.py`,
`jobs/infrastructure/worker.py:63-101`, `document/application/document_job_enqueue.py`.

**Problem.** `CONTEXT.md:262` states the invariant plainly: the user tracks one Product job,
and when inference reports back once the platform merges the result and marks it done. The
code enforcing that is spread across five files, with the reasoning carried in long comments
rather than in a module.

**[reported]** Three unrelated paths create Product jobs, one of which (`record_local_job`)
writes a job already `done`, skipping the state machine. Two independent drivers run the same
sweeps in the same order for the same reason, and the ordering constraint is documented twice.

**Solution.** Concentrate the transitions (enqueue, claim, delegate, complete, fail, time out,
cancel, record-already-done) in one module, and let the worker, the callback route, and the
on-read sweep all drive it.

**Benefits.** *Locality:* the scattered comments become one file's documentation.
*Leverage:* callers stop needing to know about claim fencing or sweep ordering.
*Testability:* the state machine becomes testable as transitions, without Postgres or the two
live uvicorn servers `tests/nomicous/integration/ml/conftest.py` currently boots.

Do this after candidate 3, which removes the storage-level duplication first.

### 8. Give the Registry a lifetime

**Files:** `inference/registry/__init__.py`, `registry/resolve.py`,
`inference/helper/routes/info.py:97`, `inference/jobs/runner.py:157`,
`nomicous/backend/ml/api/registry.py:24-26`, `nomicous/frontend/src/inference/registry.ts:19`.

**Problem.** The Registry is a file format that every caller parses for itself, not a module.
`registry.yaml` is re-read and re-parsed on every `/info` and `/run` request. There is no
reload policy and no observable version, even though the helper already computes an ETag in
`registry_sync.py`.

**[reported]** The `registry://<id>?tag=<tag>` reference format is parsed independently in
Python (`inference_dispatcher.py:34-39`) and in TypeScript
(`frontend/src/inference/registry.ts:19`). Registry path resolution is duplicated four ways.
`resolve.py` is a 24-line pass-through that returns the model entry, so both callers still
have to index by tag themselves.

**Solution.** One owned load-and-cache with explicit invalidation, and a lookup returning the
version entry the caller actually wanted. The Registry module should own the `registry://`
format so Python and TypeScript stop each inventing a parser.

### 9. Decide Host eligibility in Python

**Files:** `inference/contracts/common.py:41`, `inference/registry/__init__.py:58`,
`inference/api/jobs.py`, `nomicous/frontend/src/inference/helperInfo.ts:105-145`.

**Problem.** **[verified]** Host eligibility is a first-class domain term with a Python enum,
a registry field, and a `/info` route exposing it, and it is read by zero backend Python. The
local-versus-remote routing decision lives entirely in frontend TypeScript.

The hosted inference API will therefore run a model declared `local`, and every model in
`registry.yaml` is currently marked `local`.

**Solution.** Make Host eligibility a check the inference API and the platform dispatcher both
perform against the registry entry they already resolve. The browser keeps the Inference
preference; the backend keeps the eligibility.

**Benefits.** *Locality:* "which model, on which host" answered in one place instead of three,
one of which is TypeScript. *Leverage:* a Server-tier model can be added without a frontend
change. *Testability:* eligibility becomes unit-testable in Python.

### 10. Type the platform API seam from the generated schema

**Files:** `nomicous/frontend/src/api/client.ts:59-186`.

**Problem.** Roughly 19 wire types are hand-written beside generated equivalents in
`schema.d.ts`, bypassing a codegen path that is already wired (`codegen:api` plus a
`check:api` staleness guard).

**[verified]** The drift is already live. `text_source` has zero occurrences in backend Python
and zero in `openapi.json`, yet `client.ts:51` declares it and `hooks/utils.ts:76` branches on
`text_source === "model"`. That condition is always false against real API data. Only test
fixtures set it, so the tests pass while the production path is dead.

**[reported]** Also drifted: `SegmentPartRequest` is missing `model_id`;
`LineResponse.line_transcriptions` is typed required while the schema says optional, and is
dereferenced unguarded at `utils.ts:16`; `points` is narrowed from `number[][]` to
`[number, number][]` with no runtime check, and all canvas geometry rests on that narrowing.

**Solution.** Derive from the generated schema. Where the generated shape is too loose to be
useful, narrow it in one named place that states it is a narrowing, the way
`inference/helperInfo.ts:27-63` already handles its boundary.

**Benefits.** *Leverage:* `codegen:api` starts catching drift instead of being shadowed.
*Testability:* narrowing becomes a testable parse rather than a compile-time claim. Expect an
initial burst of type errors; each one is a latent runtime bug.

## Decisions that are yours, not the code's

### The Processing pipeline is documented, not speculative

**[verified]** `annotation/application/processing.py` is a pluggable N-step pipeline
(`SUPPORTED_STEPS`, `StepCallback`, `apply_step`, `process`) serving exactly one step, and
`process()` has zero callers. Read as code, it is textbook speculative abstraction.

But `nomicous/CONTEXT.md:178` states "Processing logic is pluggable; v1 UI/backend must not
assume it is finished", and the glossary names `normalize_height` and `binarize` as future
steps. The fence has a reason recorded in the domain model rather than in the file.

The unreachable `raise` at `processing.py:54` is noise either way. Line 50 already raises for
anything outside `SUPPORTED_STEPS`, and line 52 handles the only member.

### Glossary terms with no implementation

Several `CONTEXT.md` terms have no code behind them. Either they are roadmap, or the glossary
asserts invariants the system does not have.

- **Export** and **Export state**, described in `CONTEXT.md:186-193` as "what matters to the
  user", have zero frontend presence. No export button, no dirty indicator, no unpaired
  warning.
- **Segment overlap** is defined precisely with a worked dialogue example and a hard
  relationship rule at `CONTEXT.md:140`. **[verified]** It is enforced nowhere in either
  language. The only `overlap` matches in the codebase are BLLA decoder internals.
- **Annotation history** and **History snapshot** have backend endpoints in `schema.d.ts` and
  no client method or UI.
- **Human review** and **Review status** have a client method called only from
  `DocumentDetailPage`, not from the editor where reviewing happens.
- **Kraken ceiling** ships on every `LineResponse` and is read by no frontend code. The
  specified **Kraken ceiling overlay** does not exist.
- **Page transcription** and **Text line** have a wire contract, server state, hooks, and
  tests, but the only UI is a `visually-hidden` block in `PageEditorToolbar.tsx:179-191`.
  `api.importPageTranscription` has no production caller.

The last one is worth calling out: hidden-only UI that tests exercise converts test coverage
into false confidence.

### `package://` is a documented scheme with no users

**[reported]** The `package://` weights source branch in `inference/weights/__init__.py:40-53`
has zero registry entries, zero tests, and zero callers, yet it is a first-class term in
`inference/CONTEXT.md:34`. Either delete it and the vocabulary entry, or implement it. One
adapter is a hypothetical seam.

## Leave these alone

Verified as deep, deliberate, or load-bearing.

- `canvasGeometry.ts`. Real geometry behind small pure functions, well tested, no React. The
  model for the rest of the frontend.
- `verify_artifact_sha256` and its memo (`artifacts.py:26-65`), with
  `tests/hf/test_artifacts.py` covering memoization, same-size swaps, and mtime rewrites.
- `fetchWithAuthRecovery` and `apiRequest` (`client.ts:216-368`). Single-flight refresh,
  one-retry-then-redirect, CSRF, in-flight GET dedup.
- `inference/helperInfo.ts` and `inference/preference.ts`. Both validate their boundary and
  state why.
- `_reject_fully_failed_batch` (`calamari/adapter.py:54-68`), tied to the 503/422 mapping.
- Both Dockerfiles. They diverge deliberately and document why.
- The `_ExportGroupNorm` two-stage reduction in `blla/export.py:16-57`, carrying 25 lines
  explaining a real onnxruntime float32 accumulation bug.
- Parity-critical numerics in `blla_decoder/lines.py`, `polygon.py`, and
  `calamari/preprocessing/`. These are line-by-line ports of reference algorithms guarded by
  `test_blla_parity.py`. Renaming locals is safe; reordering operations is not.
- **The lazy `__getattr__` imports in `blla/__init__.py:8-35` and
  `calamari/__init__.py:14-29`.** These are deliberate torch avoidance for the frozen helper.
  **[verified]** `torch` and `safetensors` are in the `export` dependency group only, absent
  from `inference`, `helper`, `train`, and `platform-prod`. Making these imports eager breaks
  the signed bundle.

## Cleanup backlog

Tactical items that do not need architectural decisions. Grouped by area, roughly in
value order within each group.

### Correctness

- **[verified]** `tests/inference/unit/test_blla.py::test_standalone_helper_returns_onnx_blla_response_for_real_image`
  fails with `assert 503 == 200`. The sibling test at line 263 guards on
  `skipif(not BLLA_ONNX_ARTIFACT.is_file())`; this one does not.
- **[verified]** `scripts/platform/seed_dev_annotated_data.py:80` reads an environment
  variable literally named `"Greek Data Given by professor chitwood"`. The name and default
  value were swapped. Ruff's SIM112 would catch it, but `pyproject.toml:201` suppresses SIM112
  for `scripts/**`, so the ignore list is masking a real bug.
- **[reported]** `estebanData.py:28-33` wraps `dict.get` in `try/except KeyError`, which never
  raises. A missing filename returns `None` where the caller expects a string.
- **[reported]** `.pe-model-preview` is defined twice in `page-editor.css` (lines 1094 and
  1206). The first wins, silently disabling the second block's layout rules.
- **[reported]** `PageEditorCanvas.tsx:473-490` has `"suppressBaselineSegmentId"` twice in one
  `Omit<>` key list.
- **[reported]** `PageEditorCanvas.tsx:617,662` miss `commitPendingVertexEdit` in their
  dependency arrays. These are the only two real hook warnings in the tree.

### Dead code

**[verified]** Each of these occurs exactly once in the repository, at its own definition:

- `nomicous/backend/core/exceptions.py:28` `DatabaseUnavailableError`. Never raised, caught,
  or handled, while all five siblings are mapped in `core/app.py:149-245`.
- `nomicous/backend/document/application/part_service.py:31` `list_parts`.
- `nomicous/backend/users/application/auth_service.py:84` `get_user`.
- `nomicous/frontend/hooks/useOCR.ts`. Tracked in git, 0 bytes, outside `tsconfig`'s
  `include` so never typechecked. Flagged independently by three audits.

**[reported]** Also dead:

- `inference/architectures/calamari/{config,layers,model}.py`. 41 lines of pure re-export
  shims with zero importers, kept alive only so `packaging/helper/excludes.txt:47-49` and
  `verify-bundle.py:28-30` have names to exclude.
- `useLayoutMutations.ts` exports `moveSelectedSegmentRight`, `canUndo`, `canRedo`, and
  `editUndoRevision` with no consumer. `editUndoRevision` is a `useState` whose only job is
  forcing re-renders for `canUndo`/`canRedo`, which nothing reads.
- `INFERENCE_CALAMARI_NUM_PROCESSES` appears in 4 compose blocks and is read by no Python
  code. **[verified]**
- `src/api/client.ts:205` `API_ORIGIN`, plus two stale `vi.mock` factory properties
  referencing it.
- `@ant-design/icons`, `eslint-config-next` (also misfiled under `dependencies` so it ships
  to production), and `eslint-plugin-react-refresh` in `nomicous/frontend/package.json`.
- Six unused landing screenshots (~47 KB) and two dead `.shots` CSS blocks.
- Directories containing only stale `__pycache__`: `nomicous/backend/scripts/`,
  `nomicous/scripts/`, `tests/inference/live/`.

### Duplication

- `_is_placeholder_secret` exists in 3 copies and `_validate_service_url` in 2, across
  `inference/infrastructure/settings.py` and `nomicous/backend/core/settings/{ml,auth}.py`.
  Two different functions are both named `get_inference_settings` and return different types.
- `_file_fingerprint` is copy-pasted 4 times across `inference/architectures/`.
- **[reported]** `userFacingMessage()` in `src/api/userFacingError.ts:24` is used in one page
  while 30 sites across 15 files hand-roll a worse version.
- **[reported]** `sortByOrder` is reimplemented 12 times across 9 files.
- **[reported]** `PageEditorStatusAlerts.tsx:42-65` has eight `useEffect`s of identical shape.
- **[reported]** `BackgroundJobsContext.tsx` has four copies of one job-patch updater;
  `usePageEditorData.ts:191-286` has four copies of one `Promise.allSettled` handler.
- **[reported]** `src/train/calamari/train.py` and `finetune.py` are roughly 85% identical
  with four real deltas. The three Hydra configs copy-paste their `hydra:` and `wandb:` blocks
  verbatim.
- **[reported]** `_png_bytes` is defined 4 times with 4 signatures in tests; two are
  byte-identical. `tests/fixtures/paths.py` exists but 8 files re-derive `REPO_ROOT` anyway.
- **[reported]** The `HELPER_REGISTRY_URL` validation block is byte-identical across the macOS
  and Linux installers. It guards against `sed` template injection, so drift there is a
  security risk.
- **[reported]** Every image is Pillow-decoded and `verify()`'d twice per request, once at the
  route and once in `run_model`, on manuscript scans up to 100 MB, using two different
  settings objects.

### Lint, format, and types

- **[verified]** 533 violations sit behind `per-file-ignores`; `pyproject.toml` records 519,
  so it is drifting upward. 370 are auto-fixable.
- **[verified]** 93 of 456 files would be reformatted. CI runs `ruff check .` but never
  `ruff format --check`, and pre-commit's format hook is path-gated to a subset. Python under
  `src/`, `packaging/`, `deploy/`, and `scripts/hf/` is linted but never format-checked
  anywhere, which is why `scripts/hf/` is split between 2-space and 4-space indentation.
- **[reported]** 385 of the recorded lint violations are in `src/model/calamari/`, a vendored
  upstream package. Adding it to `extend-exclude` drops the total from 533 to about 148 and
  deletes the largest ignore entry. Worth noting in the `pyproject.toml` comment so nobody
  tries to pay down upstream's debt.
- **[reported]** No Python type checker exists anywhere. The 13 `# type: ignore` comments are
  verified by nothing. The frontend, by contrast, runs `tsc --noEmit` in CI and passes clean.
- **[reported]** `tests/inference/integration/test_blla_parity.py` (453 lines, 34% of all
  suppressions in the repository) has never run. Its `parity` dependency group is installed
  nowhere. Same for `load-testing` and `train`.
- **[reported]** No test imports `src/train`, `src/model/kraken`, or `src/preprocessing_data`.
  A three-line import smoke test would have caught the broken `finetuning.py` import on the
  day it landed.

### Git and repository hygiene

- **[reported]** `.git` is 96 MB packed, and 43% of that is one file: `src/experiments/trOCR.ipynb`
  at 41.63 MB of embedded cell outputs, still that size at `HEAD`. The pending `nbstripout`
  pre-commit change fixes it going forward (the worktree copy is already 16 KB), but the blob
  stays in history unless rewritten. Two other notebooks are fat in both `HEAD` and the
  worktree.
- **[reported]** Roughly 22 MB of model weights are tracked live under `src/hf/staging/` and
  `src/hf/local/`. `blla.onnx` is in the current diff, so every re-export permanently appends
  about 5 MB. `.gitignore` just gained `src/hf/cache/` with a comment saying weights "must
  never be committed"; that policy is not applied to `staging/` or `local/`.
- **[reported]** Three nested `.gitignore` files use bare `.env*` with no `!*.example`
  negation, so new template files there are silently unaddable. Existing ones survive only
  because tracked files bypass ignore rules.
- **[reported]** `.pytest_cache/`, `.mypy_cache/`, and `.ruff_cache/` are not ignored at root.
- **[reported]** gitleaks runs in CI but not in pre-commit, so a secret is caught only after
  it is pushed.
- **[reported]** One historical secret: `infrastructure/.env` committed in `e81a50c`
  (2026-05-21) and removed in `784df29`, still reachable. Assessed as placeholder-grade
  (localhost URLs, a `change*`-style `JWT_SECRET`, no high-entropy values). Rotate
  `JWT_SECRET` only if that placeholder ever reached a deployed environment.
  **[verified]** No `.env` file is tracked, and the working tree has no matches for `sk-`,
  `AKIA`, `ghp_`, or `hf_`.
- **[reported]** Roughly 1.5 GB of reclaimable local disk, all correctly gitignored: 216 MB
  turbopack cache, about 700 MB of DMGs and PyInstaller output.
- **[reported]** `origin/vision-transformer` is at an identical SHA to `origin/main`;
  `feat/bugfix-polish-batch` is merged; 3 `copilot/fix-*` bot branches are open; 10 stashes
  exist including two from long-dead detached-HEAD work.
- **[verified]** `AGENTS.md` is tracked but deleted in the working tree, with no `CLAUDE.md`
  or `.claude/` replacement. The repository currently has no agent-context file. The docs
  audit found zero dangling references to it.

### Documentation

The prose is not typical AI slop. **[reported]** Zero hits for "seamless", "robust",
"comprehensive", "leverage"; zero trailing `, ensuring...` clauses; zero signposting; no
filler sections.

The actual signature is different: **[verified]** only 10 em dashes survive across tracked
markdown while ` - ` appears as a mid-sentence dash substitute in the dozens (26 lines in
`nomicous/CONTEXT.md` alone). The newer untracked docs kept their em dashes. That reads as a
post-hoc de-AI-ification pass, and the result is more mechanical than the original.

The real documentation problem is staleness:

- **[verified]** Six live references to the deleted `/inference/v1/catalog` route:
  `README.md:137`, `packaging/helper/README.md:73,162`,
  `docs/guides/using-and-hosting.md:122`, `docs/inference/adding-inference-models.md:249,269`.
  A test now asserts that route returns 404.
- **[verified]** `HELPER_SECURE_MODE` is documented in `packaging/helper/README.md` and does
  not exist in source. `tests/inference/unit/test_helper_app.py:232` explicitly asserts that a
  "leftover `HELPER_SECURE_MODE` from an old install must not lock users out".
- **[reported]** Kraken is presented as the live segmenter across 9 files, while
  `docs/security/vex-click:17` and `docs/todo.md:3` already say it is gone. The corpus
  contradicts itself.
- **[verified]** `nomicous/backend/README.md:235` documents `GET /auth/me`; the router has no
  prefix and is mounted bare, so the route is `GET /me`.
- **[reported]** 8 broken relative links, all off-by-one `../`, in
  `docs/deployment/production.md` and `scripts/hf/README.md`. Documented rate limit is 60
  against a code default of 10; JWT expiry 60 against 15.
- **[reported]** `nomicous/README.md:14-23` documents `NOMICOUS_DATA_ROOT`, which no Python
  code reads. `nomicous/backend/.env.example` declares five `NOMICOUS_*` variables that
  nothing reads.
- **[reported]** The Docker Compose quick start exists in 4 near-verbatim copies, all sharing
  the same missing-secret bug. The "start the API" incantation exists in 5 copies across 2
  mutually inconsistent styles. Env-var tables exist in 5 overlapping versions, none complete,
  one mutually contradictory.
- **[reported]** `docs/adr/0001` is untracked while `docs/database-design.md:229` already
  links to it, so that reference breaks for anyone cloning.
- **[verified]** `todo.md` and `docs/todo.md` are different documents that share a filename.
  The root one is a dated punch-list for the in-flight refactor; the `docs/` one is a standing
  roadmap. The collision is the problem, not the content.

### Prose worth rewriting

**[reported]** One file carries essentially all the AI-writing tells in the source tree:
`nomicous/backend/ml/application/device_service.py`. Its `start_pairing` docstring spends 13
of 16 lines narrating a rate limit that was removed, which is a commit message living in an
API docstring. It also carries aphorisms ("Deleting it beats defending it"), reST section
headings in a module docstring, and the only hedge-y user-facing error string in the
repository.

Everything else scored clean. Do not spend cleanup budget hunting for more.

## Corrections

Two claims from the audit passes did not survive verification. Recorded so they are not
re-derived.

**The two job repositories are not against different databases.** An early structural pass
described them as "two genuinely different job queues" and called the duplication
"architecturally justified". **[verified]** They run against the same Postgres with identical
SQL predicates. See [candidate 3](#3-one-lease-primitive-two-adapters).

**`src/` is not a legacy tree.** An audit prompt framed it as possibly abandoned. It is live
and integral to model development and training. Not being imported by `nomicous/` or
`inference/` is the expected architecture: `src/` is the research and training lifecycle
stage that feeds them. Only `src/hf/resolve` and `src/hf/paths.py` cross into the runtime.

Two further framing notes. `src/model/calamari/` (96 of `src/`'s 132 Python files) is a
vendored upstream package; excluding it, `src/` is 36 files and 3,781 lines, and functions
over 50 lines drop from 61 to 29. And a transient `F821 Undefined name _run_merge` appeared in
one `ruff check` run and was not reproducible; `_run_merge` appears nowhere in the codebase.
There is no undefined-name bug.
