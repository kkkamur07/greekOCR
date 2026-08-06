# Test suite review, 2026-08-05

A critical pass over every test in the repository, asking one question of each: **would
this fail if the behaviour it names broke?** Produced by six parallel audits (two over
`tests/nomicous/unit`, one over `tests/nomicous/integration`, one over `tests/inference` +
`tests/hf`, two over the frontend suite), each required to verify its claims against
production source before reporting them.

Scope at the time of review: 878 Python tests across `tests/nomicous`, `tests/inference`,
`tests/hf`, plus 182 frontend tests in 45 files. Line numbers are accurate as of commit
`b952448`. Treat file paths as durable and line numbers as hints — and note that an
`inference/` → `export/` refactor has since begun, so the inference line numbers below are
the most likely to have drifted.

Claims are marked:

- **[verified]** — re-checked directly against the working tree by running something.
- **[mutation-verified]** — the audit copied the source into a scratch tree, broke the
  production code, re-ran the suite, and observed that it still passed. The strongest
  evidence here; the scratch copies were deleted and the real tree still passed.
- **[reported]** — the audit read the test and the production code it covers and reasoned
  about the gap, but nothing was executed to prove it.

## Status of remediation

Work is happening on the branch `test/suite-hardening`, per the plan agreed after this
review. Applied so far:

| Item | State |
| --- | --- |
| `test_process_one_job_runs_every_stale_sweep` reaching a live database | **Fixed.** All four sweeps are patched; the ordering assertion now includes the device-lease sweep and each sweep's deadline kwarg. Mutation-checked: moving the lease sweep after the claim release turns it red. |
| `test_grayscale_helper_is_the_only_convention_under_src` (red) | **Deleted.** The executing parity test covers the property; the scan only failed on a spelling. |
| `convert("L")`, `weights_only=True`, safetensors, and cuda/mps source greps | **Deleted.** Each had an executing test beside it that proves the same property strictly better. |
| Frontend auth mocking, live-Postgres worker sweeps, remaining grep replacements | Planned, not yet applied. |

---

## Contents

- [The diagnosis](#the-diagnosis)
- [Do these first](#do-these-first)
- [Tests that cannot fail](#tests-that-cannot-fail)
- [Tests that assert text instead of behaviour](#tests-that-assert-text-instead-of-behaviour)
- [Tests that mock what they are testing](#tests-that-mock-what-they-are-testing)
- [Weak assertions](#weak-assertions)
- [Coverage gaps](#coverage-gaps)
- [Leave these alone](#leave-these-alone)
- [Method and corrections](#method-and-corrections)

## The diagnosis

This suite is well above average. Integration tests run against a real Postgres and the
real ASGI app. Both model checkpoints are committed as plain files and real forward passes
execute on every pull request. The device-pairing, claim, and lease tests mint credentials
by running the real protocol rather than writing rows, and assert both the allowed and the
forbidden side. Several files are genuinely exemplary and are listed under
[Leave these alone](#leave-these-alone).

The problem is not sloppiness. It is that **a green build proves less than it appears to
in four specific places**, and the four have nothing to do with each other:

1. A large body of the best tests in the repository never executes in CI, because a
   fixture is pinned to a developer's laptop.
2. The frontend's authentication lifecycle is mocked by every one of its consumers, so the
   real implementation is executed by no test at all.
3. Roughly fifteen tests assert on the *text* of a source file rather than on behaviour.
   They pass on a comment and fail on a reformat.
4. Real model weights run constantly, but nothing asserts what the model actually *says* —
   every ML assertion is structural.

## Do these first

### 1. Thirty-eight inference CLI tests never run in CI — accepted, no action

**[verified]** `_psql` shells out to `docker exec nomicous-db-1` — a container name that
comes from `docker-compose.yml` and exists only on a developer's machine. In GitHub
Actions the Postgres is a `services:` container with a runner-generated name, so the
`docker exec` returns non-zero and the fixture calls `pytest.skip`.

- `tests/inference/integration/test_cli_run.py:52,84`
- `tests/inference/integration/test_cli_pairing.py:43,64`
- `tests/inference/integration/test_cli_self_upgrade.py:53,191`

Running those three modules with `docker` absent gives **34 skipped, 4 deselected, exit
code 0**. The nightly `ml-real-weights` job collects 18 tests out of 878, and the 4
real-weights CLI tests are exactly the ones that skip.

**Decision: accepted as-is.** These are deliberately local-only and slow to run. Recorded
here so the next reader does not mistake the green nightly for coverage of the CLI.

### 2. `test_process_one_job_runs_every_stale_sweep` reached a live database — fixed

**[verified]** It patched four of the five sweeps `process_one_job` performs. The fifth,
`release_expired_device_leases`, was left unpatched and opened a real Postgres connection
— from the suite CI runs with **no database at all**. Its ordering assertion also omitted
the lease sweep, which production runs between "waiting" and "claims"
(`nomicous/backend/jobs/infrastructure/worker.py:91`).

Fixed on `test/suite-hardening`: all four sweeps patched, ordering asserted as
`[reclaim, waiting, lease, claims]`, and each sweep's deadline kwarg asserted. The unit
lane now passes with no database reachable.

### 3. `test_grayscale_helper_is_the_only_convention_under_src` was red — deleted

**[verified]** It text-scanned all of `src/` for `COLOR_*2GRAY` against a hardcoded path
allowlist, and failed on `src/models/trocr/augmentation/weather.py:150`, where the
grayscale is a luminance intermediate inside a fog composite that never becomes a model
input tensor. The only answer to that false positive was to grow the allowlist; extending
the same scan to `inference/` would have needed two more entries on identical terms.

`test_training_and_serving_produce_the_same_model_input` already runs both real
implementations over eight PIL modes and compares the tensors. It fails on a real skew;
the scan only failed on a spelling. Deleted, with the reasoning recorded in the file.

### 4. The frontend auth lifecycle is executed by no test

**[mutation-verified]** `AuthProvider.establish` and `AuthProvider.logout` were both
replaced with empty bodies and **the entire 182-test suite still passed**.
`src/auth/AuthProvider.test.tsx` covers only the bootstrap-refresh effect, and every
consumer — `LoginPage.test.tsx`, `ProtectedRoute.test.tsx` — does
`vi.mock("../auth/AuthProvider")` and stubs `useAuthSession`.

So a sign-out that no longer calls `clearAccessToken()`, leaving a live JWT in memory and
a populated `queryClient` for the next user of the tab, is invisible to CI. So is a login
that stores no token.

### 5. The only true numerical parity test has never run

**[reported]** `tests/inference/unit/test_blla.py:152,162` compare native BLLA logits
against Kraken's reference `blla.mlmodel` at `rtol=1e-5`, guarded by
`pytest.importorskip("kraken")`. `kraken` appears in **no** dependency group, is absent
from `uv.lock`, and no CI job installs it — and `test_architecture_contract.py:401`
actively asserts kraken must *not* be importable in the inference graph.

## Tests that cannot fail

| Test | Why it cannot fail |
| --- | --- |
| `tests/nomicous/unit/test_document_access.py:68` | **[reported]** Never asserts a raise. The two `require_can_read(...)` calls are bare statements; the only assertion is `issubclass(AccessDeniedError, Exception)`, true for any exception class with the production code deleted. The branch it names is also unreachable — `can_read_document` returns `True` whenever `workflow == published`, so the `raise` is dead code the test papers over. |
| `src/auth/session.test.ts:19` | **[mutation-verified]** Deleting `loginRedirectInFlight` from `beginLoginRedirect` left the test passing. It asserts `router.replace` was called once, but `redirectToLogin` calls `window.location.assign` — the assertion cannot observe the second caller. |
| `src/components/ProtectedRoute.test.tsx:27` | **[mutation-verified]** The call site was changed to pass a dead router, so no navigation could occur, and the test passed. Both the hook and the navigation are stubbed, and the assertion is a bare `toHaveBeenCalled()`. |
| `src/components/AuthenticatedImage.test.tsx:19` | **[mutation-verified]** `isPermittedOrigin` was replaced with a bare origin equality — deleting the guard against fetching protected page images over plaintext http — and both this file and `imageCache.test.ts` passed. It mocks `API_BASE_URL` on `../api/client`, which `imageCache.ts` never reads. |
| `src/components/public/PublicPageCanvas.test.tsx:17` | **[mutation-verified]** Deselection was removed entirely and the test passed. `selectedRegionId` is a fixed `null` prop, so both key presses hit the same branch. |
| `src/api/client.auth-recovery.test.ts` (refresh-rejection branch) | **[mutation-verified]** `redirectToLogin()` was removed from the catch arm and the full suite produced zero new failures. |
| `tests/nomicous/unit/test_job_lifecycle.py:568` | **[reported]** Asserts `"pg_try_advisory_xact_lock" in inspect.getsource(...)` — and that exact string appears in the module docstring at `stale_sweep.py:22`. The assertion is satisfied by prose. |
| `PageEditorPlaceholderPage.transcription.test.tsx:514` | **[reported]** Its distinguishing assertion is true by construction: the fixture selects the ground-truth layer, and the strip hard-codes the label to "Save". No `text_source` logic exists in the page-editor source. |
| `tests/nomicous/unit/test_public_boundary_hardening.py:50` | **[reported]** Asserts `page.items == [cursor]` on a Pydantic model the test just constructed. |
| `tests/nomicous/unit/test_line_geometry.py:45` | **[reported]** Compares `polygon` to itself — `copied` was defined as `{"points": polygon}` two lines earlier. |
| `tests/nomicous/unit/test_media_store.py:43` | **[reported]** `absolute_path` raises unless the path is under the root, so the assertion can never be `False`. |
| `tests/nomicous/unit/test_job_lifecycle.py:800` | **[reported]** Ends with `ClearJobHistoryResponse(deleted=3).model_dump() == {"deleted": 3}` — a test of Pydantic on a value the test constructed. |

## Tests that assert text instead of behaviour

They pass on a comment, a docstring, or a dead branch, and fail on a reformat that changes
nothing. Ranked by what is at stake:

- **`tests/nomicous/unit/test_destructive_script_guards.py:71`** — **[reported]** greps six
  bash literals out of `scripts/platform/reset_supabase_nonprod.sh`, whose payload is
  `DROP TABLE ... CASCADE` against a live database. Changing `exit 1` to `exit 0` inside
  the non-production guard, or breaking `env_file_value` so `"true "` parses as empty,
  keeps every assertion passing. The script honours `SUPABASE_ENV_FILE` and invokes `psql`
  bare, so it can be run in a sandbox with a stub `psql` on `PATH`.
- **`tests/nomicous/unit/test_deployment_hardening.py:159`** — **[reported]**
  `assert "USER appuser" in dockerfile`. The last `USER` wins; append `USER root` and the
  container runs as root with the test green. CI already proves the real property by
  running `docker run --entrypoint id` against the built image.
- **`test_deployment_hardening.py:203`** — **[reported]** asserts a path *the test
  constructs* is a file, then greps `fonts.py` for two strings. Point `_ASSETS_FONT` at a
  typo and both still match while every transcription PDF raises at runtime.
- **`test_deployment_hardening.py:196`** — **[reported]** greps `build.sh` for two `copy2`
  literals, when `test_platform_bundle.py` already runs the script into a tmpdir and
  asserts the exact resulting file tree.
- **`test_deployment_hardening.py:275`** — **[reported]** asserts `"NOLOGIN"` appears once
  (satisfied by construction) and four role names appear somewhere. Adding
  `GRANT ALL PRIVILEGES ... TO nomicous_platform_worker` destroys the read-only boundary
  the test's name defends and passes.
- **`tests/nomicous/unit/test_device_lease_wiring.py:73,93,140`** — **[reported]** three
  tests grep function bodies. `:93` checks a literal appears in each caller's source —
  wrap the call in `if False:` and the sweep stops running while the test stays green.
- **`tests/nomicous/unit/test_device_pairing.py:242`** — **[reported]** the constant-time
  claim rests on `"hmac.compare_digest" in inspect.getsource(...)`. Constant-time is not
  assertable from Python, and `secret_matches` computes the HMAC *before* comparing, so
  the compare's timing is unobservable even in principle. The four behavioural assertions
  beside it are good and should stay.
- **`tests/nomicous/unit/test_job_lifecycle.py:641,858`** — **[reported]** the
  `asyncio.to_thread(...)` call-shape grep, and a `downgrade()` asserted by substring while
  `test_migrations.py` already reverses it for real.

Deleted during remediation: the cuda/mps grep at `test_architecture_contract.py:419`
(superseded by `test_torch_runtime.py:174`, which loads both real checkpoints and asserts
every parameter's `device.type`), the `weights_only=True` substring at
`test_torch_runtime.py:249` (superseded by the `__reduce__` payload test twenty lines
above, whose digest matches so only `weights_only=True` can make it pass), the safetensors
substring half, and the two grayscale greps.

## Tests that mock what they are testing

- **`tests/nomicous/integration/test_jobs.py:499`** — **[reported]** monkeypatches
  `notify_platform_job_status_changed` inside the suite whose whole point is end-to-end
  realism. That function issues a real `pg_notify` against the Postgres this suite already
  has. Its real body swallows every exception, so a channel-name mismatch kills every live
  job-status update in production and this test still passes.
- **`tests/nomicous/unit/test_media_compensation.py:70`** — **[reported]** the most
  consequential single-test problem found. The fake repository appends to a list; the real
  `enqueue_media_deletion_intent` does `session.add(...)` then `session.commit()` **on the
  very session whose commit is raising**. On the exact failure the test names, the intent
  commit also raises, the inner `except` swallows it, and the orphaned WebP leaks forever.
- **`tests/nomicous/integration/test_ml_job_callback.py:525,561`** — **[reported]** patches
  a private symbol of the module under test to inject a fault. The rollback asserted is
  real, but the fault is injected at the wrong seam.
- **`src/components/projects/ProjectJobsPanel.test.tsx:18`** — **[reported]**
  `vi.mock("../../hooks/useJobPolling")` stubs the panel's entire live-update contract.
  **[mutation-verified]** separately, `load` was changed to append instead of replace and
  the test at `:45` passed — it asserts the API was called, never that the table changed.
- **`tests/nomicous/unit/test_document_access.py:16-33`** — **[reported]** builds
  `MagicMock()` rows for a pure predicate over plain ORM objects. Renaming
  `Project.shared_users` without updating `is_member` passes; the seam tests next door,
  which build real rows, would fail.

## Weak assertions

**HTTP** — **[reported]** `assert status_code in (403, 404)` appears throughout
`test_projects.py:139,158,165,168,175`, `test_documents.py:130,137`,
`test_annotation_history.py:254-256`, `test_pairing_progress.py:336`,
`test_export_approved_line_artifacts.py:133`, `test_transcription_pdf_artifact.py:149,222`,
`test_jobs.py:835`. Accepting either code means the test cannot distinguish "authorization
denied" from "route does not exist". In the two export tests the URL literal is re-typed
rather than shared with the happy path, so a typo'd path 404s and the test passes having
proven nothing. `test_pairing_progress.py:337` and `test_access_public.py:118` do this
correctly — they re-read the resource, or assert the owner's 200 on the same URL.

Also: `test_jobs.py:149` asserts only `status in ("pending","running","done")` and no
timestamp despite its name; `test_access_public.py:138` asserts an empty layout because the
fixture creates no lines; `test_transcription_pdf_artifact.py:198` asserts a `<Baseline>`
byte-identical to the `<Coords>` two lines above, enshrining the fallback rather than the
baseline — and `_geometry_points` reads only `geometry["points"]` while kraken baselines
are persisted as GeoJSON `coordinates`.

**Frontend** — **[reported]** `ProjectsPage.test.tsx:48` and `ProjectDashboardPage.test.tsx:79`
assert the delete API was called, never that the row disappears; `DocumentDetailPage.test.tsx:118,143`
stub `getDocument` with a constant fixture so the badge could never flip;
`transcription.test.tsx:286` stubs `"fresh ocr"` and never asserts it renders;
`segmentMutations.test.tsx:150` asserts a call satisfied by the initial page load;
`PublicDocumentPage.test.tsx:225` asserts "geometry" as `"Regions: 1"` rendered by a mock;
`ModelOutputBlock.test.tsx:30-32` queries by CSS class for a property the adjacent
`getByTitle` assertions already prove.

**Structure-only** — **[reported]** `test_signed_page_image_wiring.py:111` asserts only
`inspect.signature` for both media-store backends, and `:126` — the test covering the
Supabase signing path that runs in production — is an unconditional `@pytest.mark.skip`
whose body raises. `test_execution_target.py:153` restates the private `_ELIGIBLE_TARGETS`
as a literal, so changing `eligible_targets_for_model` to `return ALL_EXECUTION_TARGETS`
leaves it green.

## Coverage gaps

1. **Collaborator write access is untested on every route below the project level.**
   **[reported]** `document/domain/access.py:16` gates all 31 document, part, line, history,
   export, and job routes on `is_member`, and a shared user is a full member — so a
   collaborator has write access to all of them. The `collaborator_headers` fixture is
   never passed to any of those routes.
2. **The model-binding surface has no authorization test of any kind.** **[reported]** All
   ten project-scoped routes in `ml/api/models.py` are exercised with a single
   `auth_headers` fixture.
3. **Nothing asserts what the model says.** **[reported]** `_decode_greedy` has no direct
   test, no golden transcription exists for the Syriac fixture line, and
   `test_calamari_pytorch.py:55` — whose name claims parity with the vendored Calamari
   processors — never imports a vendored processor; it asserts a tensor shape. The hand
   reimplementation of `center_normalize`/`_dewarp` is the highest train/serve skew surface
   in the codebase and that is its only coverage.
4. **The live SSE push path is untested end to end.** **[reported]** `test_jobs.py:269`
   polls the job to `done` before opening the stream, so it only reads the snapshot frame.
5. **Destructive document routes have no test at all.** **[reported]**
   `POST .../layout/reset` and `POST .../copy-to-ground-truth` return zero grep hits.
6. **The pairing *deny* path is untested.** **[reported]** Approve, expiry, wrong-code, and
   double-redeem are covered thoroughly; denial is not.
7. **Optimistic rollback for segment mutations.** **[reported]** No test rejects
   `patchPartLine`/`deletePartLine`. The Accept / promote-to-ground-truth path is stubbed
   and never called.
8. **`src/utils/jobSubscription.test.ts` covers 2 paths of a 226-line module.**
   **[reported]** Both drive the polling fallback; the entire SSE success path is untouched.
9. **Smaller:** **[reported]** `test_project_access.py` tests only the ownerless branches;
   `test_line_geometry.py` never exercises the GeoJSON `coordinates` branch production
   takes; `test_media_store_factory.py` covers only `local` while production runs Supabase;
   `LoginPage`/`RegisterPage` have no failure-path test.

**Process note.** **[reported]** `nomicous/frontend/tsconfig.json` excludes
`**/*.test.ts(x)` from `tsc --noEmit`, so fixtures drift from the generated contract:
`jobSubscription.test.ts` and `transcription.test.tsx` build job responses with
`payload`/`user_id`/`document_part_id` while `executionTarget.test.ts` uses
`project_id`/`part_id`/`execution_target` — mutually incompatible shapes, both green.

## Leave these alone

Named explicitly so nobody "improves" them, and because they are the pattern the rest
should follow.

**Python** — `tests/nomicous/integration/test_device_claim.py`, `test_device_lease.py`,
`test_execution_target.py`, `test_agent_version_floor.py`, `test_signed_page_image_link.py`
are the best work in the repository: they mint credentials by running the real pairing
protocol, make time-dependent behaviour deterministic by writing the production signal
(`updated_at`, `last_seen_at`) instead of patching a clock, and assert both sides of every
boundary. `test_migrations.py` is the single highest-value test here.
`tests/inference/unit/test_torch_runtime.py` observes `torch.is_inference_mode_enabled()`
from *inside* the graph and proves digest verification is ordered ahead of `torch.load`
with a `__reduce__` payload. Also strong: `test_document_catalog.py`,
`test_document_access_seam.py`, `test_agent_version.py`, `test_document_part_dimensions.py`,
`test_platform_bundle.py`, `test_ground_truth_text.py`, `test_replace_part_lines_defaults.py`,
`test_runtime_configuration_security.py`, `test_media_encoding.py`,
`test_document_job_enqueue.py`, `test_document_patch_semantics.py`, and
`test_published_package.py`.

**Frontend** — `src/api/client.auth-recovery.test.ts` stubs `fetch` at the network boundary
only and proves one shared refresh across concurrent 401s, the refreshed token on the
retry's header, and independent abort signals. `src/pages/page-editor-placeholder/testSupport.tsx`
renders the real page inside the real provider and mocks only the api client;
`executionTarget.test.tsx` is the strongest file, mounting twice against a stateful
preference stub. Also clean: `segmentNumbering`, `imageCache`, `FormModal`, `canvasGeometry`,
`editUndo`, `characterConfidence`, `cursorPagination`, `getCache`, `userFacingError`,
`publicLayout`.

## Method and corrections

Six audits ran in parallel, each given the same rubric and instructed to verify against
production source before reporting, to treat mocking of a true external boundary (S3, the
Hub, a GPU model) as *correct*, and to name the files that are genuinely good rather than
manufacture findings for them. One audit additionally verified its most important claims by
mutation testing in a scratch copy.

Two claims raised during the review were checked and found not to be problems:

- An audit reported "unrelated Python whitespace-only modifications" staged in the working
  tree. **[verified]** `git diff --cached` was empty and `git update-index --refresh`
  cleared the entries: a stale index-stat artifact from running the suite, not a change.
- A suspicion that the frontend's "does not name a host" tests were vacuous because
  `expect(null).not.toMatch()` might silently pass. The audit checked empirically that this
  *fails* in Vitest 4, and withdrew the finding.

Finally, `tests/inference/unit/test_schema.py:8` is misfiled: it lives under
`tests/inference` but imports `infrastructure.alembic.versions.*`, and despite its name it
never enumerates the versions directory, so it cannot detect the two-heads condition it
claims to guard. Asserting `ScriptDirectory.from_config(...).get_heads()` has length 1
would.
