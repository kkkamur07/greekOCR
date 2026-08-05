# Merge handoff — codebase review remediation session (2026-08-04)

## 1. Branch topology: only ONE branch actually needs merging

Trunk is **`feat/inference-cli-redesign` @ a0bc170**. It contains *all* of this session's work.

| Branch | Worktree | Ahead | Behind | Action |
|---|---|---|---|---|
| `feat/inference-cli-redesign` | repo root | — | — | **trunk** |
| `feat/049-torch-runtime-archive-onnx` | `.claude/worktrees/agent-af7f62c4…` (locked) | 0 | 0 | identical — nothing to do |
| `feat/frontend-libraries` | `greekOCR-frontend` | 0 | 1 | fast-forward |
| `feat/049-torch-free-runtime-boundary` | `.claude/worktrees/agent-a5bcce9f…` | 0 | 1 | fast-forward |
| `feat/deep-cleanup` | `greekOCR-deepclean` | 0 | 3 | fast-forward |
| **`feat/048-collapse-second-job-queue`** | `.claude/worktrees/agent-ac976295…` (locked) | **4** | **4** | **real merge** |

Five of six are fast-forwards or no-ops. All merge risk lives in `feat/048-collapse-second-job-queue`,
which branched from `21c24b2` — *before* the device-pairing commit (`0b998ec`) and before every commit
in this session.

Its 4 commits:
```
99ca23f refactor(platform): delete the HTTP hop that fed the second queue
c2a5ff0 refactor(inference): delete the inference service job queue
4b1262f chore: remove legacy configs directory
516c3fc feat: restore vision transformer source and configuration
```
It is implementing **ADR 0003** (single job queue), so its direction is intended, not stale.

---

## 2. The conflict surface: 28 files touched by both 048 and trunk

```
.env.compose.example                        inference/Dockerfile
.github/workflows/deployment.yml            inference/jobs/runner.py
.github/workflows/quality.yml               nomicous/backend/core/.env.production.example
.pre-commit-config.yaml                     nomicous/backend/core/.env.supabase.example
inference/admission.py                      nomicous/backend/core/app.py
inference/api/run.py                        nomicous/backend/core/settings/ml.py
nomicous/backend/document/application/local_inference_service.py
nomicous/backend/jobs/application/job_callback_service.py
nomicous/backend/jobs/infrastructure/worker.py
nomicous/infrastructure/alembic/versions/001_initial_schema.py
pyproject.toml                              scripts/platform/reset_supabase_nonprod.sh
uv.lock
tests/inference/integration/test_jobs_worker.py   tests/inference/integration/test_run.py
tests/inference/unit/test_admission.py            tests/inference/unit/test_blla.py
tests/nomicous/integration/conftest.py            tests/nomicous/integration/test_jobs.py
tests/nomicous/integration/test_ml_job_callback.py
tests/nomicous/unit/test_deployment_hardening.py
tests/nomicous/unit/test_runtime_configuration_security.py
```

### HIGHEST RISK — resolve these deliberately, do not auto-merge

**`nomicous/infrastructure/alembic/versions/001_initial_schema.py`**
Trunk **froze this migration into explicit literal DDL**. It previously called
`Base.metadata.create_all(bind=bind)`, which meant CI structurally could not detect a missing
migration (head *was* the ORM, so a diff could never fail). It now creates 18 tables, 33 indexes
(incl. one GIN and four partial), and 9 enum types by hand, reconstructing the **pre-002** state.
- **Never regenerate this file from live ORM metadata.** That is the defect that was fixed.
- 048 also edits this file (it deletes the `inference_jobs` queue). The correct resolution is to keep
  trunk's frozen DDL and have 048's removal expressed as a **new migration**, not by editing 001.
  Rewriting 001 breaks any database already stamped past it.
- Guard test: `tests/nomicous/integration/test_migrations.py::test_migration_chain_matches_orm_metadata`
  migrates a scratch DB to head and asserts the autogenerate diff is empty. It was negative-controlled
  (a phantom column made it fail), so trust it. **If it goes red after the merge, the merge is wrong.**

**`nomicous/backend/jobs/application/job_callback_service.py`**
Both sides restructure this heavily. Trunk invariants that must survive:
1. **Claim is its own committed transaction**; then **verify → merge → finalize is ONE transaction with
   exactly one `commit()`**, holding `SELECT … FOR UPDATE` on the job across the merge. Previously the
   document merge and the job status committed separately, so a crash between them left merged lines
   under a `failed` job.
2. `notify_platform_job_status_changed` fires **outside** the session, after commit, on all three paths.
3. `_public_callback_error()` redacts URLs/paths/tokens before anything reaches `job.error` — that field
   is rendered directly to users (`JobsNotice.tsx:59`). Raw text goes to the log only.
4. `_apply_transcribe_merge_sync` **skips lines where `output is None`** and records
   `failed_line_indexes`; a batch with zero usable lines raises rather than merging silently.

**`inference/jobs/runner.py` + `inference/admission.py`**
- Per-line error isolation: one failing line must not discard the page; if **every** line fails, the
  **first original exception is re-raised** so a broken artifact stays 503 and a bad request stays 422.
- `SEGMENT_PARAM_BOUNDS` / `validate_segment_params` live in `admission.py` deliberately — it is the one
  seam every entry point crosses (sync `/inference/v1/run`, the queued job path, the local helper).
  Do not relocate into the runtime; a bound on the request model alone is bypassable.
- `inference/architectures/{artifact,isolation}.py` are shared seams added by trunk with 26 contract
  tests (`tests/inference/unit/test_architecture_contract.py`) run over all four execution paths and
  mutation-verified. Keep them.

**`.github/workflows/quality.yml`**
- `ruff format --check` must keep its **explicit path list**
  (`nomicous/backend inference scripts/platform tests/nomicous tests/inference tests/hf`), NOT `.`.
  It must stay identical to the `files:` pattern on the `ruff-format` pre-commit hook, which
  deliberately excludes `src/` (vendored Calamari fork). If they diverge, CI rejects what the hook produces.
- New `ml-parity` job runs `-m ml` against the tracked weights; gated to non-PR events.

**`scripts/platform/reset_supabase_nonprod.sh`** — guards now parse the env file **before** merging it
into the environment, and require a typed target. An exported `SUPABASE_NON_PRODUCTION=true` must NOT
satisfy them (that was the documented-but-broken workflow). `docs/deployment/supabase.md` was rewritten to match.

### Lower risk
`pyproject.toml` / `uv.lock` — trunk adds `zxcvbn>=4.5.0` to `platform-prod`. Re-run `uv sync` after
merging; CI uses `--locked`.
`nomicous/backend/document/application/local_inference_service.py` — trunk makes the persisted `Job.id`
the single id used for **both** the provenance stamp and the returned `result["job_id"]` (they used to
be two different UUIDs, making local-run provenance unqueryable).

---

## 3. Other trunk invariants a merge must not silently revert

- **`nomicous/backend/users/api/rate_limit.py`** — the `Content-Type` gate on identity extraction is
  **removed entirely** (casefolding alone still left `application/*+json` and a missing header as holes).
  Oversized bodies return **413** rather than falling through unthrottled (pydantic `extra='ignore'` made
  a padded body a valid login that skipped identity extraction). Unattributable requests are charged to a
  coarse `unattributable:<path>` bucket; the old fail-open `return` is gone.
- **`nomicous/backend/core/settings/auth.py`** — JWT secret strength uses the **`zxcvbn` library**, gated
  on `guesses_log10 >= 22.0`. It must NOT use zxcvbn's `score`: that saturates at 4, which
  `correcthorsebatterystaple1234567` clears. Measured gap: memorable ≤16.9, worst of 5000
  `token_hex(16)` draws 27.45.
- **`nomicous/backend/core/settings/_cache.py`** — `settings_cache` enrolls accessors at decoration time;
  `reset_settings_caches()` clears all of them. Three hand-written clear-lists existed and **two were
  missing accessors**, leaking storage/device settings between tests. Do not reintroduce a manual list.
- **`media_store/encoding.py`** — pixel bound is checked from the image **header before decode**; the
  process-wide `Image.MAX_IMAGE_PIXELS` mutation and `warnings.catch_warnings()` are gone (both unsafe
  under `asyncio.to_thread`). Encoding happens *outside* the decode guard so encoder `OSError` is a server
  error, not a 422.
- **Public thumbnails** — width is an allowlist (`PublicThumbnailWidth` = 200/400/800), results are cached,
  and requests are throttled. `tests/load/locustfile.py` was updated from `w=1200` to `w=800`.
- **Frontend** — `runLocalFirstWrite({cloudEnabled, runLocally, runInCloud})` has **no `refresh` parameter
  and must never gain one**. That is what makes "a failing cosmetic read triggers a billed cloud run"
  structurally impossible. Two canary tests guard it:
  `usePairingState > keeps a persisted local result when the reload that follows it fails` and the
  matching `useLayoutMutations` test. **If either is deleted or weakened during a merge, stop.**
- **`inference/architectures/calamari/preprocessing/conversion.py`** — uses `image.convert("L")` to match
  the serving path. The old channel-count dispatch fed palette PNGs' *indices* to the model as luminance
  (98.6% of pixels differ, up to 231 levels); CMYK and I;16 likewise; LA raised. Guarded by
  `tests/inference/unit/test_calamari_grayscale_parity.py`.
- **`packaging/helper/scripts/verify-bundle.py` `FORBIDDEN_MODULE_PREFIXES`** is pinned by
  `tests/nomicous/unit/test_deployment_hardening.py:79`. Renaming any module named there silently
  vacates the Torch-leak bundle check. 048 touches both files — check this explicitly.
- **`.gitignore:83` → `/.claude/`** — keeps ephemeral agent worktrees (full repo checkouts) out of
  `git add -A` and out of `ruff check .`.

---

## 4. Known-failing tests — pre-existing, do NOT chase or "fix" during the merge

| Test | Cause |
|---|---|
| `tests/inference/unit/test_blla.py::test_standalone_helper_returns_onnx_blla_response_for_real_image` | 503, model weights unavailable in this environment |
| `tests/nomicous/integration/test_device_pairing.py` (5 tests) | asyncpg "attached to a different loop" — that module builds its own `TestClient` in a second event loop and collides with the session-scoped client |
| `tests/nomicous/integration/test_ml_job_callback.py::test_callback_unconfigured_secret_returns_503` | a gitignored root `.env` supplies `INFERENCE_WEBHOOK_SECRET`, so `monkeypatch.delenv` cannot unconfigure it. Passes in CI. |
| `tests/nomicous/integration/test_documents.py::test_member_create_list_read_update_delete_document` | errors **only** when run alongside `test_device_pairing.py`; passes in isolation |

---

## 5. Verification after merging — expected numbers

Postgres for integration: `127.0.0.1:5433`, user `postgres`, password `dev`, db `kalamos`.

```bash
# Unit — expect 600 passed, 1 failed (the BLLA 503), 3 skipped
JWT_SECRET=test-secret-not-for-production-at-least-32-bytes \
  uv run --group test --group platform --group inference --group export \
  pytest tests/nomicous/unit tests/inference/unit tests/hf -q

# Integration — expect 147 passed + exactly the known failures above
JWT_SECRET=test-secret-not-for-production-at-least-32-bytes \
INFERENCE_WEBHOOK_SECRET=test-inference-webhook-secret \
INFERENCE_SERVICE_SECRET=test-inference-webhook-secret \
  uv run --group test --group platform --group inference \
  pytest tests/nomicous/integration -m "not ml" -q

# Frontend — expect 49 files / 205 passed, typecheck clean,
# lint clean except 3 known warnings (PageEditorCanvas.tsx x2, LoginPage.test.tsx x1)
cd nomicous/frontend && npm run typecheck && npm run lint && npx vitest run

# Gates — both must be clean
uv run --group dev ruff check .
uv run --group dev ruff format --check nomicous/backend inference scripts/platform \
  tests/nomicous tests/inference tests/hf
```

Note 048 deletes the inference job queue, so its own integration tests
(`tests/inference/integration/test_jobs_*.py`, `tests/nomicous/integration/test_ml_job_callback.py`)
legitimately change shape. Expect counts to move there — but the *unit* count should only ever go up,
and the four known failures above should stay exactly four.

---

## 6. Two latent defects found but deliberately NOT fixed

Both in `document_catalog.update_document`, both currently unreachable through HTTP. Worth an issue,
not a merge-time change:

1. **`workflow=None` writes NULL to a non-nullable column.** The guard reads
   `if "workflow" in fields and fields["workflow"] is not None`, so an explicit `None` skips *both* the
   type check and the owner check. Only `DocumentUpdateRequest.reject_explicit_null` prevents it.
2. **`**fields` shadows the positional parameters.** A request naming `session`, `user`, `project_id`,
   or `document_id` raises `TypeError` (500) instead of the `ValidationError` (422) the whitelist gives.
