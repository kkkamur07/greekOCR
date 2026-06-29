# Future fixes — post-027 code review

> Source: second-pass five-axis review of `feat/027-remove-root-app-duplicates` (2026-06-18).
> Parent context: `issues/done/027-remove-root-app-duplicates.md`, `issues/prd-annote-merge.md`.

This document tracks **remaining** work after the 027 fix pass. Items marked **Resolved** are kept for audit trail only — do not re-open unless regressions appear.

---

## Summary

| Severity | Open | Resolved |
|----------|-----:|---------:|
| Critical | 2 | 2 |
| Required | 4 | 9 |
| Important | 7 | 1 |
| Optional / Nit / FYI | 12 | — |

**Merge blockers today:** Critical #1–2. Required #3–6 strongly recommended before merge.

---

## Resolved in 027 fix pass

| ID | Issue | Evidence |
|----|-------|----------|
| R1 | XFF rate-limit bypass | `rate_limit.py` gates `X-Forwarded-For` on `behind_proxy` + trusted proxy peer |
| R2 | Re-pairing left stale ground truth | `document_service.py:491-499`; `test_repairing_candidate_text_line_clears_previous_segment_ground_truth` |
| R3 | Re-import page transcription didn't clear GT | `document_service.py:437-444`; `test_reimport_page_transcription_clears_stale_pairing_ground_truth` |
| R4 | History restore omitted `block_id` | `history_service.py:169`; `test_restore_snapshot_preserves_line_block_associations` |
| R5 | `paired_line_count` semantics mismatch | `history_service.py:49-53` uses stripped non-empty `approved_text` |
| R6 | Public layout API stub (backend) | `public.py:70-82` + `list_document_layout_public` |
| R7 | Transcribe jobs tagged `"test": true` | `document_service.py:590` — payload is `{"adapter": ...}` only |
| R8 | Pipeline tests polled jobs without auth | `_poll_job(..., headers=...)` in segment/transcribe pipeline tests |
| R9 | Broken import in `test_transcribe_pipeline.py` | Uses `backend.tests.platform.test_documents` |
| R10 | `ENABLE_TEST_JOB_ROUTES` unset in conftest | `conftest.py:16` |
| R11 | `errors.ts` dropped validation `details` | Parses `error.details` with `loc` + `msg` |
| R12 | Stale `annote.main` entrypoint in pyproject | Removed; Docker uses `backend.core.main:app` |
| R13 | Migration `011` undocumented | Listed in `infrastructure/README.md` |

---

## Critical — blocks merge

### F1. Root pytest `pythonpath` breaks platform test imports

**Files:** `pyproject.toml:57`, `annote/README.md`, `annote/backend/README.md`

**Problem:** `pythonpath = ["."]` points at repo root, but `backend` and `infrastructure` live under `annote/`. `uv run pytest annote/backend/tests/platform/` fails with `ModuleNotFoundError: No module named 'infrastructure'` unless `PYTHONPATH=annote` is set manually.

**Fix:**
- Set `pythonpath = ["annote"]` in root `pyproject.toml` (or equivalent path that exposes both packages).
- Align all README test commands so they work without manual env vars.

**Acceptance criteria:**
- [ ] `uv run pytest annote/backend/tests/platform/test_root.py` imports successfully from repo root (Postgres errors OK without DB).
- [ ] `cd annote && uv run --project .. --group platform pytest backend/tests/platform/test_root.py` imports successfully.
- [ ] README snippets match the working command.

---

### F2. Background worker races repo-level SKIP LOCKED claim tests

**Files:** `annote/backend/tests/platform/test_jobs.py:158-233`, `annote/backend/core/app.py:151`, `annote/backend/jobs/infrastructure/worker.py:110-117`

**Problem:** Session-scoped `TestClient` starts `worker_loop`. With default `JOB_WORKER_CLAIM_TEST_ONLY=None`, the worker claims synthetic pending jobs inserted in `test_claim_next_uses_skip_locked` and `test_concurrent_claimers_do_not_run_same_job_twice`, marks them failed, and races the test assertions.

**Fix (pick one):**
- Separate `TestClient` fixture without lifespan worker for repo-level claim tests.
- Pause/disable worker in `test_jobs.py` autouse fixture during claim tests.
- Restructure: pipeline tests call `process_one_job()` explicitly; integration client fixture omits worker.

**Acceptance criteria:**
- [ ] `test_claim_next_uses_skip_locked` and `test_concurrent_claimers_do_not_run_same_job_twice` pass reliably with lifespan worker enabled elsewhere.
- [ ] Segment/transcribe pipeline tests still pass (worker must process non-test jobs).

---

## Required — fix before merge

### F3. No test asserts standardized 429 error envelope

**Files:** `annote/backend/tests/platform/test_auth.py:108-124`, `annote/backend/core/app.py:55`, `annote/backend/core/schemas/errors.py`

**Problem:** Rate-limit tests assert status code only. Backend returns `{ error: { code: "RATE_LIMITED", message: "...", details? } }` and `Retry-After` header — untested regression risk. 401 envelope is tested; 429 should match.

**Fix:** Add assertions on `response.json()["error"]["code"] == "RATE_LIMITED"`, message, and `Retry-After` header in rate-limit tests.

**Acceptance criteria:**
- [ ] Login rate-limit test asserts full error envelope.
- [ ] Register rate-limit test added (see F14).

---

### F4. `test_root` hardcodes version string

**Files:** `annote/backend/tests/platform/test_root.py:10`, `annote/backend/core/version.py`

**Problem:** Test pins `"0.3.2"`; fails on every `VERSION` file bump.

**Fix:** Import `get_version()` and assert `body["version"] == get_version()`.

**Acceptance criteria:**
- [ ] Version test survives `annote/VERSION` bumps without edit.

---

### F5. Duplicate `AuthService` singletons

**Files:** `annote/backend/users/api/auth.py:16`, `annote/backend/users/api/dependencies.py:19`

**Problem:** Two module-level `AuthService()` instances, each with its own `UserRepository`. Harmless today (stateless) but splits DI and complicates future mocking.

**Fix:** Single shared instance via `get_auth_service()` FastAPI dependency or one module-level singleton imported by both.

**Acceptance criteria:**
- [ ] One `AuthService` construction path for auth routes and `get_current_user`.

---

### F6. `BEHIND_PROXY=true` without `FORWARDED_ALLOW_IPS` only warns

**Files:** `annote/backend/core/app.py:161-165`, `annote/backend/core/settings/app.py:20-21`

**Problem:** Misconfigured proxy trust logs a warning but app starts. Operators may believe client IPs come from XFF when they do not.

**Fix:** `model_validator` on `AppSettings` that fails startup when `behind_proxy=true` and `forwarded_allow_ips` is empty/unset (or fail unconditionally in non-dev).

**Acceptance criteria:**
- [ ] Invalid proxy config prevents startup or raises at settings load.
- [ ] Valid config (`behind_proxy=false` or allow-list set) still starts.

---

## Important — strongly recommended

### F7. `PublicDocumentPage` fetches layout but renders empty canvas/panel

**Files:** `annote/frontend/src/pages/PublicDocumentPage.tsx:43-46, 78-79, 185-206`

**Problem:** API loads `layout` and `layers`, but `regions` and `transcriptions` are hardcoded `[]`. Public view shows image only — no segments or transcription text despite backend data.

**Fix:** Map `PublicLayoutResponse.lines` → `Region[]` and transcription layers → `Transcription[]` (mirror `PageEditorPlaceholderPage` or `DocumentDetailPage` mapping).

**Acceptance criteria:**
- [ ] Published document with layout shows segment overlays on `ImageCanvas`.
- [ ] Transcription panel shows ground-truth (or published layer) text for selected segment.
- [ ] Integration or component test covers non-empty public layout.

---

### F8. Legacy frontend orphan code breaks test/typecheck graph

**Files:** `annote/frontend/src/app/`, `src/lib/api.ts`, `PageEditor.tsx`, `PageCard.tsx`, `PageCard.test.tsx`, etc.

**Problem:** Next.js-style modules unreachable from `App.tsx` but still in vitest/tsconfig scope. `npm test`: 16/17 files pass, `PageCard.test.tsx` fails (`next/link`). `npm run typecheck` fails. `npm run build` passes (tree-shaken).

**Fix:** Delete orphan tree or exclude from `vitest.config.ts` / `tsconfig` includes. Update stale README references.

**Acceptance criteria:**
- [ ] `npm test` — all files pass.
- [ ] `npm run typecheck` passes (or legacy dirs explicitly excluded with documented reason).
- [ ] `annote/frontend/README.md` no longer references `LegacyDemoApp` / `services/`.

---

### F9. Duplicate conflicting type declarations in `types/index.ts`

**Files:** `annote/frontend/src/types/index.ts` (duplicate `DrawMode` / `EditMode` blocks)

**Problem:** `EditMode` declared as `'none' | 'vertices'` and `'none' | 'move' | 'vertices'` — breaks strict typecheck.

**Fix:** Single canonical declaration; align with `ImageCanvas` usage.

**Acceptance criteria:**
- [ ] One `DrawMode` and one `EditMode` export.
- [ ] `npm run typecheck` no longer fails on this file.

---

### F10. History restore incomplete for full annotation state

**Files:** `annote/backend/annotation/application/history_service.py:158-177`, `document_service.py:556-562`

**Problem:** Snapshots omit `baseline`, `mask`, `manual_geometry`. Restore rebuilds them from `points`/`source` — custom baselines/masks lost. Restore also does not reconcile `PageTranscriptionLine.paired_line_id` (GT text restored but pairing metadata may disagree).

**Fix:** Extend snapshot schema; restore pairings or document explicit semantics.

**Acceptance criteria:**
- [ ] Snapshot round-trip preserves baseline/mask/manual_geometry OR docs state intentional loss.
- [ ] Restore + pairing UI stay consistent (paired_line_id reconciled or cleared).

---

### F11. PATCH line cannot unset `block_id`

**Files:** `annote/backend/document/application/document_service.py` (line update path ~371-373)

**Problem:** Update skips `None` values — `{"block_id": null}` silently ignored. Create/restore handle `block_id`; PATCH does not.

**Fix:** Treat explicit `null` as unassign block.

**Acceptance criteria:**
- [ ] PATCH with `block_id: null` clears block association.
- [ ] Regression test added.

---

### F12. Dead code: `ml/application/adapters.py`

**Files:** `annote/backend/ml/application/adapters.py`

**Problem:** `ModelAdapterRegistry` and noop stubs have zero imports. Worker uses `kraken_adapter.py` directly.

**Fix:** Remove file or wire registry when ML bindings connect to job execution.

**Acceptance criteria:**
- [ ] No orphaned adapter registry, or registry used by worker/model service.

---

### F13. Duplicate job-creation paths (drift risk)

**Files:** `annote/backend/jobs/application/job_service.py`, `jobs/infrastructure/job_repository.py`, `document/application/document_service.py:597-617`

**Problem:** `JobService.enqueue_segment_job` mirrors `DocumentService.enqueue_segment_part` but only the latter is wired from HTTP. Paths may diverge on `model_id`/`binding_id`.

**Fix:** Route all enqueue through `JobService` or delete unused service methods.

**Acceptance criteria:**
- [ ] Single enqueue path for segment/transcribe jobs from API layer.

---

## Optional / Consider

### F14. Register endpoint rate limiting untested

**Files:** `annote/backend/users/api/auth.py:22`, `test_auth.py`

Add 429 envelope test for `POST /auth/register` (mirror login tests).

---

### F15. `FORWARDED_ALLOW_IPS=*` re-enables XFF spoofing

**Files:** `annote/backend/users/api/rate_limit.py:30`

Wildcard trusts XFF from any direct client. Reject `*` at settings validation or document as dev-only.

---

### F16. Rate-limit TOCTOU race

**Files:** `annote/backend/users/api/rate_limit.py:65-80`

Check-then-insert without row lock allows brief burst above limit. Acceptable for auth; use atomic increment if strict enforcement needed.

---

### F17. Rate limit counts successful auth attempts

**Files:** `rate_limit.py`, `auth.py`

Successful login/register consume quota. Consider counting failures only or separate limits.

---

### F18. Auth rate-limit tests lack `@pytest.mark.integration`

**Files:** `test_auth.py:108+`

Rate-limit tests hit Postgres but are unmarked unlike sibling auth tests.

---

### F19. No test for `behind_proxy=true` + untrusted peer ignores XFF

**Files:** `test_auth.py`

Add case: `BEHIND_PROXY=true`, peer not in allow-list → falls back to `request.client.host`.

---

### F20. `get_version()` no fallback if `VERSION` file missing

**Files:** `annote/backend/core/version.py:9-10`

Missing file → uncaught `FileNotFoundError`. Add default or clear error at startup.

---

### F21. `processing.py` has no unit tests

**Files:** `annote/backend/annotation/application/processing.py`

Legacy `test_polygon.py` / `test_rectify.py` deleted with monolith. Add unit tests for rectify/export geometry path.

---

### F22. Positive integration test for public layout with geometry

**Files:** `test_access_public.py:123-130`

Only asserts empty layout. Add fixture with blocks/lines on published document.

---

### F23. `TranscriptionDetail.tsx` missing icon imports

**Files:** `annote/frontend/src/components/TrascriptionPanel/TranscriptionDetail.tsx:113,155`

`EditOutlined` / `ThunderboltOutlined` referenced but not imported — typecheck failure.

---

### F24. No frontend unit tests for `parseApiError` / `error.details`

**Files:** `annote/frontend/src/api/errors.ts`

Regression risk for recently fixed validation detail parsing.

---

### F25. `vite.config.ts` lacks `@/` path alias

**Files:** `annote/frontend/vite.config.ts` vs `vitest.config.ts`

Vitest has alias; Vite build does not. Add for consistency if legacy imports ever enter production graph.

---

### F26. Root `.gitignore` has no global `.env` pattern

Relies on `annote/backend/.gitignore`. Consider `**/.env` at repo root.

---

### F27. `App.tsx` token check not reactive

`isAuthed` reads token once; in-place clear without navigation won't update shell. Low impact (401 redirect handles it).

---

## FYI — track, no immediate action

| Topic | Notes |
|-------|-------|
| **Branding** | App title `"greekOCR Platform"` vs root `"Kalamos API"` — align or document intentional split |
| **`get_job` 403 vs 404** | Ownerless jobs return 403 (existence leak); consider 404 |
| **Job visibility** | Enqueuer-only; collaborators can't poll another member's job |
| **Worker ignores bindings** | `model_id`/`binding_id` not used at execution — expected until OCR design (`028`) |
| **No job reaper** | Crashed worker leaves jobs in `running` — pre-existing pattern |
| **Export response size** | Unbounded base64 JPEGs in single JSON — pagination/streaming later |
| **`replace_part_lines` N+1** | Per-line `_block_or_404` with 10k cap — batch-load blocks |
| **History POST returns full state** | Large JSONB on create — consider summary-only response |
| **Public layout N+1** | Two queries per part — fine for small docs |
| **Empty-string `approved_text`** | Writes empty GT row instead of deleting; progress ignores whitespace |
| **`paired_line_count` naming** | Counts GT-with-text, not pairing links — name may mislead API consumers |
| **Duplicate GIN index** | ORM + migration both declare `ix_jobs_payload_gin` — autogenerate noise |
| **`JOB_WORKER_CLAIM_TEST_ONLY` undocumented** | In `.env.example` but not `backend/README.md` settings table |
| **`infrastructure/README.md` volume name** | May not match compose project prefix for `docker volume rm` |
| **`clear_auth_rate_limit_state` naming** | Documented no-op — confusing name |
| **`DatabaseUnavailableError` unused** | `core/exceptions.py` — health uses inline SQLAlchemy catch |
| **Stale rate-limit rows** | Only pruned per-key on hit — periodic global TTL cleanup optional |
| **CORS headers narrowed** | Confirm frontend never sends headers beyond `Authorization`, `Content-Type` |
| **JWT secret validation** | Rejects placeholder but not short secrets — enforce min length in prod |
| **`test_transcribe_pipeline.py` marker** | Missing `@pytest.mark.integration` vs siblings |

---

## Suggested implementation order

1. **F1** — PYTHONPATH (unblocks all CI/local pytest)
2. **F2** — Worker/test isolation (unblocks reliable job tests)
3. **F3, F4** — Quick test hardening
4. **F7, F8, F9** — Frontend merge readiness
5. **F5, F6** — Auth/proxy polish
6. **F10–F13** — Domain completeness (can ship as follow-up PRs)
7. **F14–F27** — As capacity allows

---

## Verification checklist (when fixes land)

- [ ] `uv run pytest annote/backend/tests/platform/` — green with Postgres
- [ ] `npm test` + `npm run typecheck` — green in `annote/frontend`
- [ ] `npm run build` — still green
- [ ] Public published document shows layout + transcription in browser
- [ ] Rate-limit tests assert `RATE_LIMITED` envelope
- [ ] No regressions on pairing/GT/history fixes (R1–R13)
