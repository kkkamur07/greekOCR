# Merge audit — `feat/inference-cli-redesign` (103 commits, `21c24b2..5d62a35`)

Eight parallel reviewers, five-axis rubric (correctness / readability / architecture / security /
performance) plus a dedicated dead-code and simplification sweep. Repo-root `src/` excluded
throughout. Every finding below was verified against the file on disk by the reviewer that
raised it; findings that could not be proven were dropped.

Scope reviewed: `inference/`, `nomicous/backend/`, `nomicous/frontend/`, `nomicous/infrastructure/`,
`tests/`, `docs/`, `issues/`, `config/`, `deploy/`, `scripts/`, `.github/workflows/`, build and
compose files.

---

## The one systemic finding

**The comments are the specification here, and several of them now assert safety properties the
code does not deliver.** This codebase documents itself in long explanatory prose rather than in
external docs — which is a strength until the prose stops being true, at which point it is worse
than no comment, because a future reader trusts it instead of re-deriving the property.

Nine independent instances, each verified:

| File | The comment claims | The code does |
|---|---|---|
| `inference/cli/upgrade.py:36-40` | "a compromised platform cannot choose what executes here" | installs a platform-named package (§1) |
| `inference/architectures/isolation.py:3-7` | "a single degenerate contour cannot discard the other thirty-nine lines" | one degenerate contour discards the page (§6) |
| `jobs/application/job_claim_service.py:169` | streaming scans through the serverless API "was rejected (ADR 0002)" | streams the scan inline, base64, on every claim (§5) |
| `jobs/application/job_claim_service.py:79-88` | the agent prefix "can never collide with `worker_identity()`" | a host named `agent` collides (§27) |
| `users/api/rate_limit.py:270-275` | "no real sign-in ever enters this bucket" | 100% of `nomicous pair` traffic enters it (§10) |
| `core/app.py:388-390` | "each device router carries `require_device_pairing_enabled`" | `internal_inference.py` does not (§30) |
| `layout_service.py:11-13,173-174` | bulk paths "preserve kraken source" | they reset it to `manual` (§14) |
| `patch_fields.py:1-7` | an unrecognised key "has to be refused loudly" | pydantic drops it before the guard runs (§25) |
| `media_store/encoding.py:25` | 17 lines arguing `MAX_DECODE_PIXELS` is the operative guard | the branch is unreachable; Pillow raises first (§26) |
| `inference/cli/run.py:279-282` | the interrupt window is "a handful of bytecodes" | a second window discards completed work (§8) |
| `api/resources.ts` header | "a mutation quietly forgetting to refresh a list has nowhere left to live" | the entire page editor forgets (§12) |

Fix the comment or fix the code, but do not merge with both standing.

---

## BLOCKING

### 1. CRITICAL — Remote code execution via self-upgrade
`inference/cli/upgrade.py:212`

```python
requirement = f"{floor.package}>={floor.minimum_version}"
command = _installer_command(requirement)   # -> pip install --upgrade <requirement>
```

`floor.package` is taken verbatim off the wire (`api.py:553`, `api.py:643`) and never compared
against `DISTRIBUTION_NAME` — which is imported at `upgrade.py:59` and used only in printed
strings. Anyone controlling the platform response (compromised platform, MITM per §2, or a
researcher pointed at a hostile `--api-url`) returns a 426 naming `attacker-pkg`, and every agent
runs `pip install --upgrade attacker-pkg>=9.9.9` at its next launch. An sdist executes
`setup.py` at build time; a wheel declaring a `nomicous` console script replaces the entry point
that `_re_exec` then `execve`s. Code execution as the researcher, no human in the loop.

Found independently by two reviewers.

**Fix:** refuse unless `floor.package == DISTRIBUTION_NAME`; validate `minimum_version` against
`^\d+\.\d+\.\d+$` before interpolation; prefer a pinned `==` over an open `>=`.

### 2. CRITICAL — The claim loop, pairing, and upgrade are untested in CI
`tests/inference/integration/test_cli_run.py:52,86,98` (same shape in `test_cli_pairing.py`,
`test_cli_self_upgrade.py`)

```python
POSTGRES_DSN = "postgresql://postgres:dev@localhost:5433"
...
["docker", "exec", "nomicous-db-1", "psql", "-U", "postgres", "-c", sql]
...
pytest.skip(f"cannot reach Postgres at {POSTGRES_DSN}: ...")
```

`quality.yml:97` runs these against a service container on **5432** with a run-id password and
exports `DATABASE_URL`/`SYNC_DATABASE_URL`/`MIGRATOR_DATABASE_URL`. There is no `nomicous-db-1`
on a GitHub runner and no `os.environ` fallback, so ~35 of the 38 tests skip. The three headline
features of this merge are green because they never execute.

Compounding it, `test_cli_run.py:326` and `test_cli_pairing.py:229` **skip** when the built wheel
fails to install — while asserting on the build two lines earlier. A Torch dependency-resolution
regression, which the fixture docstring says is "part of what is under test", turns the suite
green.

**Fix:** read the DSN from `MIGRATOR_DATABASE_URL` with the compose value as default; create the
database via SQLAlchemy `AUTOCOMMIT` as `test_migrations.py:62-69` already does; make install
failure an assertion.

### 3. CRITICAL — `ruff check .` fails on HEAD, and CI gates on it
`pyproject.toml` `[tool.ruff.lint.per-file-ignores]` + `.github/workflows/quality.yml:48`

Verified on the committed tree: `uv run ruff check .` exits **1** with **164 errors** — 159 in
`src/models`, 3 in `src/train`, 2 in `src/inference`. `extend-exclude` covers only
`src/experiments`; `per-file-ignores` names `src/model/**` (singular) and `src/preprocessing_data/**`,
neither of which matches `src/models/**` (plural, restored by `516c3fc`).

`quality.yml:48` runs `ruff check .` with **no** `continue-on-error`, so `python-lint-unit` fails
on every pull request. `docs/resume-inference-redesign.md:70,83` records the 164 errors accurately
and calls it "a config change plus a decision about the vendored tree — not a merge task" — but the
workflow still gates on it, so the branch cannot go green. That, plus the 10 documented test
failures, means the merge is being called complete against a red build.

**Fix:** decide it explicitly — either add `src/` to `extend-exclude` (it is audit-only), add the
missing `per-file-ignores` entries as suppressions, or move `ruff check .` to `continue-on-error`
the way `mypy` already is. Do not leave it gating and failing.

---

## HIGH

### 4. No TLS enforcement anywhere in the CLI
`inference/cli/api.py:280-282`, `pair.py:99`, `run.py:132`, `upgrade.py:152-161`

`base_url.rstrip("/")` is the only handling any of these give a URL. A grep of all four modules
for `https|scheme|startswith` returns one hit: the default constant. A researcher on `http://`
(staging, self-hosted, a stale `$NOMICOUS_API_URL`) leaks the 180-day device token out of the
`X-Nomicous-Device-Token` header to anyone on-path — and turns §1 from "requires platform
compromise" into "requires the same wifi". `credentials.py` opens by naming this exact credential
as its whole threat model.

**Fix:** reject non-`https` in `PlatformClient.__init__` unless the host is loopback; refuse to
*persist* a credential whose `platform_url` is not https.

### 5. Every claim ships the full page image inline — the signed-link feature buys nothing
`jobs/application/job_claim_service.py:172`, `application/inference_dispatcher.py:80`,
`api/claim_schemas.py:59`

The claim response carries **both** `page_image_url` (signed, TTL'd) and
`request.image_bytes` — the whole scan, base64, ~1.33× size, through the serverless API.
`test_device_claim.py:182` asserts `image_bytes` is truthy, so this is shipped behaviour.
The docstring three lines above states that an authenticated image endpoint "was rejected
(ADR 0002): the production API is serverless, so streaming manuscript scans through it costs
money for nothing."

So #053 (signed page image link), `DEVICE_PAGE_IMAGE_URL_TTL_SECONDS`, the HMAC signing module,
and its integration suite are all dead weight over a path that streams the image anyway.

**Fix:** give `build_inference_submit_request` an `include_image` flag, make `image_bytes`
optional on the claim wire shape, and require the agent to use `page_image_url`. Or delete
the signed-link path and the ADR text justifying it. Pick one.

### 6. BLLA polygonization has no per-line isolation
`inference/architectures/blla/blla_decoder/__init__.py:104-127`

The decode loop calls `calculate_polygonal_environment` with no `try`. That function raises by
design — `ValueError("Invalid bounding polygon computed: ...")` at `polygon.py:192`, and
`"No intersection with boundaries"` at `polygon.py:242,254`. `blla_runtime.py:93` calls the
decoder **outside** its own `try`, which starts at line 131 and covers refinement only.

`isolation.py:3-7` states the invariant: "a single undecodable crop **or a single degenerate
contour** cannot be allowed to discard the other thirty-nine lines of the page." The degenerate
contour is raised in the one stage with no isolation.

Directly reachable: `polygon.py:105` computes `600 / (c_max - c_min)` on Python ints, so a
vertical or one-pixel-wide line environment raises `ZeroDivisionError` and takes the page with it.
`polygon.py:193` assumes the intersection is a single `Polygon` and breaks on a `MultiPolygon`.

**Fix:** wrap the per-baseline body in the same `try/except` + `first_failure` pattern
`blla_runtime.py:131-166` already uses; `max(c_max - c_min, 1)`.

### 7. One transient error kills the long-running claim loop
`inference/cli/run.py:157-159`

`_claim_loop` catches `KeyboardInterrupt` only. `PlatformError` covers `URLError`, `TimeoutError`,
`OSError` (`api.py:315-322`) and *any* non-200 (`api.py:442`), and it exits the loop. A single
502 from the serverless host terminates the agent — against `--exit-when-empty`'s own help text:
"without it, it keeps running." The one deliberate stop (426) is correct; nothing extends that
reasoning to 5xx or a DNS blip.

Aggravated by `api.py:75`: one hardcoded 30 s timeout is shared by the long-poll claim, so a
`--wait-seconds` above ~28 makes *every* claim time out client-side — while the flag's help
promises "clamped by the platform" and the platform's ceiling is settable to 120.

**Fix:** catch `PlatformError` in the loop with capped exponential backoff, re-raising
`AgentVersionRefused` and stopping on 401; pass `timeout=wait_seconds + REQUEST_TIMEOUT_SECONDS`
for the claim.

### 8. Completed transcription work is discarded and reported as failed
`inference/cli/run.py:276-294`, `:351-357`

`_handle_page` computes `output` inside its `try`, then leaves the try to print timing and call
`_report`. A Ctrl-C in that window propagates to `_claim_loop`, which has no access to `output`
and reports `status="failed"` — terminal on the platform, so the page dies with the work done.
The comment at `:279-282` asserts this window is "a handful of bytecodes between the claim
returning and `_handle_page` taking responsibility"; it is not.

Same shape at `run.py:394-396`: a `PlatformError` while posting the terminal callback is printed
and swallowed with no retry.

**Fix:** move `_report` inside `_handle_page`'s protection; retry the report on `PlatformError`
excluding 403.

### 9. `file://` and SSRF via the page-image URL
`inference/cli/api.py:465-469`

```python
request = urllib.request.Request(url, method="GET")
with urllib.request.urlopen(request, timeout=IMAGE_TIMEOUT_SECONDS) as response:
```

`url` is `str(raw["page_image_url"])`, unvalidated. The reviewer confirmed empirically that the
default opener services `file://` (read `/etc/passwd`, 9344 bytes). On a hosted worker this also
reaches `169.254.169.254`. The surrounding docstring reasons carefully about *authorization* of
this fetch and never about *where it points*.

**Fix:** require `https` (or `http` only for a loopback base URL) and assert the origin matches
`self.base_url` or the configured storage origin.

### 10. Two remote denial-of-service paths in the rate limiter
`users/api/rate_limit.py:258-264`, `:135-137`

**(a) Account lockout.** `throttle_auth_attempts` is a route *dependency*, so it charges every
attempt including successful ones, against a bucket keyed on the victim's email. Anyone who
knows an email holds that account at 429 indefinitely with garbage passwords.

**(b) Platform-wide pairing lockout.** `POST /device/v1/pairings` produces no key at all —
`attributable_client_ip` returns `None` when `TRUST_PEER_IP=false` (the documented production
posture) and `PairingStartRequest` has no `email` field — so every honest request lands in the
shared `unattributable` bucket, capped at 300/60 s. One attacker at ~5 req/s locks every
researcher out of `nomicous pair`. The comment at `:270-275` justifies the shared bucket with
"no real sign-in ever enters this bucket", which is true for `/auth/*` and false here.

**Fix:** charge the account bucket only on failure; give the pairing route its own client-specific
key.

### 11. Lease expiry re-pends forever with no attempt counter
`jobs/infrastructure/job_repository.py:287-349` (same shape at `:216`)

`release_expired_device_leases` resets `status=pending, claimed_by=NULL, ...` and reads or writes
no counter. The `Job` model has no `attempts` column — the only attempt counter in the merge is
`HelperPairing.attempts`. A page that reliably kills the agent cycles `pending → waiting → pending`
forever and never reaches a terminal status. The docstring argues correctness ("the page is still
exactly where it started") but not termination.

**Fix:** add `jobs.claim_attempts`, increment in the re-pend `UPDATE`, fail past a ceiling.

### 12. The entire page editor mutates server state and invalidates nothing
`components/page-editor/hooks/useLayoutMutations.ts`, `usePairingState.ts`,
`components/sharing/DocumentLiveSharingControls.tsx:43-46`

Neither hook imports `invalidateAfter` or `invalidateResourceTags` — a repo-wide grep finds those
only in `ProjectsPage`, `ProjectDashboardPage`, `DocumentDetailPage`. Yet `runAutoSegment`,
`replaceWithManualLine`, `updateSegmentPoints`, `deleteSelectedSegment`, `undoEdit`/`redoEdit`,
`saveGroundTruthText`, `saveApprovedText`, `promoteSelectedSegmentToGroundTruth` and both OCR
paths all write lines and transcriptions. `PublicDocumentPage` reads under
`publicDocument(projectId, documentId)` and is never invalidated by any of them.

Publishing is the same: `updateDocument({workflow})` followed only by a local `setState`.
`invalidateAfter.documentUpdated` and `invalidateAfter.projectUpdated` exist in `resources.ts`
with **zero call sites**. The module header promises "the old failure mode, where a mutation
quietly forgot to refresh a list, has nowhere left to live." It lives in the page editor.

**Fix:** add `invalidateAfter.partContentChanged(projectId, documentId)` and call it after each
committed write; wire up the two existing dead helpers.

### 13. A background refetch failure blanks a fully-loaded page
`hooks/useServerQuery.ts:141`

```ts
data: isError ? null : (query.data ?? null),
// First load only. A background refetch on window focus must not put the
// public document page back to its skeleton…
loading: key === null || query.isPending,
```

`queryClient.ts:31-34` sets `retry: false` with `refetchOnWindowFocus: true`. React Query keeps
`query.data` on a background failure; this hook throws it away. The `loading` line was
deliberately hardened against exactly this and `data` was not. The `onError` side effect fires
too, and every call site calls `toast.error` — so an offline user gets a toast on every tab focus.

**Fix:** `data: isError && query.data === undefined ? null : (query.data ?? null)`; gate `onError`
on `failureCount === 1`.

### 14. Bulk line replace silently destroys Kraken provenance
`document/application/layout_service.py:298-309`

`source_metadata`, `kraken_ceiling` and `block_id` use preserve-prior-on-absent. `source` and
`kind` do not — they fall to the schema default, and the route sends `exclude_unset=True`, which
strips it. A client omitting `source` flips every pre-existing Kraken line to
`source=manual, manual_geometry=True` **while keeping its `kraken_ceiling`** — a row that
CONTEXT.md's "Kraken ceiling constrains Auto-refine" rule no longer describes. Two comments in
the same file assert the opposite behaviour.

**Fix:** apply the preserve-prior idiom to `source` and `kind`.

### 15. A second non-prod reset crashes migration 007
`scripts/platform/reset_supabase_nonprod.sh:194-226` vs `alembic/versions/007_execution_target.py:82-90`

The drop list omits `helper_devices`, `helper_pairings`, and the `execution_target` enum.
`DROP TABLE users CASCADE` drops dependent FK *constraints*, not dependent tables, so both survive
with `inference_host` intact. `alembic_version` is dropped, so migrations replay: 005 is guarded
and skips, 007 is not — `column "inference_host" ... already exists`.

**Fix:** add the two tables and the enum (and `DROP FUNCTION jobs_execution_target_is_fixed()`)
to the drop list; guard 007's `add_column` the way 005 guards its `create_table`.

### 16. Expensive unauthenticated routes are unthrottled
`document/api/public.py:80,137-182`

Only two call sites in the repo throttle. The **cheap** one (thumbnails) is metered at 240/min
with an 18-line rationale. Unmetered and anonymous: full reportlab PDF render (with
`ImageFont.truetype` reloaded up to 41× *per line*), page-XML export, a layout read of up to
10 000 lines with nested `selectinload`, and — added by this merge — a document read that triggers
up to 25 blob downloads **and a `session.commit()`**.

**Fix:** lift the thumbnail throttle to a shared dependency across the `/public` router; cap the
anonymous layout limit well below 2000.

### 17. Blocking object-store I/O on the event loop
`document/application/part_service.py:111,117`

`self._media.write(image_key, encoded.data)` is not offloaded. `SupabaseMediaStore.write` is a
synchronous HTTPS upload of up to a 100 MiB WebP. Every other media call in the codebase *is*
offloaded — `read_part_bytes`, `_read_part_image_size`, `media_gc_loop`,
`export_service._render_segment_images` — several with docstrings saying "without blocking the
event loop". The upload path is the sole outlier and stalls every in-flight request.

**Fix:** `await asyncio.to_thread(self._media.write, ...)`; same for the compensating `delete`.

### 18. The shipped dev env template boots to a crash
`nomicous/backend/core/.env.example:30`

`JOB_WORKER_CLAIM_TEST_ONLY=` with `env_ignore_empty` unset yields `''`, not `None`, against
`bool | None` → `ValidationError`. `create_app()` calls `get_job_settings()` at `app.py:336`, so
the app dies at boot. Four documents tell the reader to copy this file. Reproduced by the
reviewer against the installed pydantic-settings.

**Fix:** comment the line out, or set `env_ignore_empty=True` in `env_settings_config()`.

### 19. Three integration tests cannot fail
- `test_device_lease.py:558-577` — `test_concurrent_agents_racing_a_swept_queue_each_get_a_distinct_page`
  passes when **zero** pages are handed out: `handed == []` satisfies `len == len(set)`, every row
  is `pending`, every `claimed_by` is `None`. The sibling at `test_device_claim.py:478` does it
  right with `assert set(claimed) == set(submitted)`.
- `test_device_claim.py:432-446` — the wait-clamp test asserts a pydantic default and a status
  code; deleting the clamp makes it hang for an hour rather than fail.
- `test_destructive_script_guards.py:71-93` — the only test of a destructive DB reset is substring
  matching over text that is never executed. A guard present but no longer terminating passes.

Also `test_torch_runtime.py:249-264`: `weights_only=True` is asserted as a substring of a file
whose *docstring* already contains the phrase, so flipping the real call at `checkpoint.py:47`
keeps it green.

---

## MEDIUM — selected

- **Webhook branch bypasses the lease check** (`jobs/api/internal_inference.py:96-107`). The
  `else` branch does no job-level authorization, and with the HTTP hop deleted nothing produces a
  job it legitimately serves. Delete it or narrow it with `job_is_held_by`.
- **Hosted workers collapse onto one device row** (`ml/application/agent_credentials.py:77,194`).
  A missing `X-Nomicous-Worker-Name` defaults to `cloud-worker`, so any cloud worker can report on
  another's leased page — breaking the invariant `agent_claim_owner`'s docstring asserts.
- **The claim query's selective predicates are unindexed** (`job_claim_service.py:108-126`). The
  only pending index is `(created_at, id) WHERE status='pending'`; the query also filters
  `type IN (...)`, `execution_target`, and `user_id`, once per agent per second.
- **`006_drop_inference_jobs` is not reversible** — `upgrade()` revokes the schema grant,
  `downgrade()` never restores it.
- **`003` and `005` justify their idempotent DDL with a claim `001` now loudly contradicts** —
  001 was frozen in this merge and says it must never be regenerated from ORM metadata.
- **Reorder takes `FOR UPDATE` but computes the shift from stale identity-map values**
  (`document_repository.py:342-355`) — needs `.populate_existing()`.
- **Unbounded ground-truth text** (`schemas.py:327`) — the only text field in the module without a
  cap. Four unbounded UUID lists reach `IN (...)` (`:117,238,243,320`).
- **Public layout silently truncates blocks with no cursor** (`document_catalog.py:181-190`).
- **`/docs` ships blank** — `CSP: default-src 'none'` blocks Swagger's CDN script, stylesheet, and
  its XHR to `/openapi.json`. CI asserts the header and never loads the page.
- **Splitting an over-merged band throws away the baseline** (`segment_refinement.py:81,89`) and
  substitutes the polygon's vertical midline; the platform's own helper uses the bottom edge.
- **Artifact preflight runs before the empty-batch check** (`calamari/adapter.py:151-160`),
  inverting the failure ordering that `artifact.py:12-24` spends a docstring on.
- **The quality gate rasterizes two full-page masks per iteration, up to 20, per line**
  (`segment_geometry.py:137`) — up to 1600 × 24 MB allocations on a large page.
- **Duplicate `focus` + `visibilitychange` listeners** (`inference/hostPreference.ts:83-100`)
  issue two concurrent requests per tab return — in the one hand-rolled fetch left in a merge
  whose purpose was to delete hand-rolled fetches.
- **`setLines` called inside a `setLayout` updater** (`useLayoutMutations.ts:148-161`) — impure,
  double-invoked in StrictMode, survives only because the operation happens to be idempotent.
- **`resetSelectedLine` has no `try/catch`** (`useLayoutMutations.ts:237`) — alone among six
  sibling mutations; all three call sites `void` the rejection, so a 403 shows nothing.
- **Repeating an operation with an identical success string produces no toast**
  (`PageEditorStatusAlerts.tsx:52-60`) — effects keyed on the message value itself.
- **`hostPreference.error` is never rendered** — a failed read is presented as the factual claim
  "using cloud", and the Retry button is hidden by the same flag that failed.
- **`config/trocr/configs.yaml:6` names a config group that does not exist** — no
  `config/trocr/output/`, so every Hydra entry point using `config_name="configs"` fails at
  composition.
- **`MEDIA_URL_SIGNING_SECRET` is the only secret with no validator** (`settings/storage.py:23`).
- **`internal_inference` router lacks `require_device_pairing_enabled`** — reachable with the
  layer nominally off, and provisions rows as a side effect.
- **Non-ASCII credential header → `TypeError` → 500** (`agent_credentials.py:176`,
  `dependencies.py:31`) — `compare_digest` on latin-1-decoded header strings. Unauthenticated
  remote 500 + traceback on the merge's most load-bearing route.
- **`deploy/platform/.env.local` is a real Vercel OIDC token at mode 0644** — untracked, correctly
  gitignored, already expired; but world-readable and there is no `.vercelignore`.
- **`packaging/` holds 1.3 GB of PyInstaller/DMG residue**, ignored only *incidentally* by the
  generic `build/` and `dist/` rules rather than a deliberate one. #061 deleted its reason to exist.
- **Pairing approval binds nothing to the requesting device, and mints a 180-day token.** The
  code says so honestly (ADR 0001 risk acceptance) — the challenge is the duration, not the design.

---

## DEAD CODE (the explicit ask)

Every entry below was proven by a repo-wide grep excluding `src/` returning only the definition.

**Backend**
- `JobRepository.record_local_job` (`job_repository.py:65`) — zero callers; orphaned by the
  deleted local-inference persist routes. Its docstring still describes them.
- `DocumentPartService.list_parts` (`part_service.py:65`) — zero callers.
- `GET /…/parts/{part_id}/transcription-pdf` (`documents.py:581`) — dead duplicate of the POST;
  the frontend and the README use only POST.
- The whole "inference service went silent" branch is unreachable: `mark_job_waiting` has no
  production caller, so `fail_stale_waiting_jobs`, `waiting_timeout_error`,
  `WAITING_TIMEOUT_ERROR`, `seconds_until_next_stale_waiting_job` and the worker's deadline
  capping all sweep a permanently empty set. `count_active_jobs` and the write-only
  `jobs.heartbeat_at` column go with them.
- `auth_service.py:66,81` — sessionless access tokens minted on every login and register, and
  discarded by both callers. They carry no `sid`, which `get_current_user` refuses outright.

**Inference**
- `JobSubmitResponse` (`contracts/jobs.py:39`) — response model of the deleted submit endpoint.
- `apply_reading_order` (`blla_decoder/lines.py:411`) — production uses `reading_order_indices`.
- `_fallback_polygon` + its guard (`blla_decoder/simple.py:74,163`) — provably unreachable.
- `blank_last_softmax` (`calamari/model.py:70`) — a full softmax computed per line, per page,
  read by nothing.
- `BLLAInput.scale_xy` — never read on the production path; the decoder recomputes it. The
  width-cap comment cites this field as the reason the cap is safe.
- Ten wire fields parsed and never read (`api.py`): `Claim.lease_seconds`, `ClaimedPage.job_type`,
  `.execution_target`, `.lease_expires_at`, `.page_image_expires_at`, `AgentFloor.reason`,
  `AgentNotice.minimum_version`/`.package`, `AgentVersionRefused.reason`/`.package`. Worst:
  `page_image_expires_at` carries a docstring explaining it exists so the agent can check the link
  — the agent fetches unconditionally and catches the 403.
- `refine_segment` (`segment_refinement.py:105`) — tests only.
- Function-level imports in `version.run` defer nothing (`main.py` imports them at module scope).

**Frontend**
- `ToastVariant` (`components/ui/toast.ts:4`) — a new file with zero references anywhere,
  including inside itself.
- `INFERENCE_HOST_NOUN`, `INFERENCE_HOST_PHRASE`, `export type { HostPreference }`,
  `AGENT_PACKAGE_NAME`, `RESOURCE_FRESH_MS`, `ResourceMeta`, `ServerQueryOptions`,
  `NumberableSegment`.
- From `useLayoutMutations`: `selectedLineSnapshot`, `moveSelectedSegmentRight`, `canUndo`,
  `canRedo`, `editUndoRevision`. The last exists only to force a re-render so `canUndo`/`canRedo`
  are re-read — with both consumers absent, that is **six pure-waste re-renders of the whole
  editor** per edit.
- `usePageEditorData`'s `initialDocument` parameter and the `canReuseDocument` short-circuit that
  serves it; the returned `setPart`.
- `client.ts` exports with zero readers: `API_ORIGIN`, `GeometryValue`,
  `UpdateLineGeometryRequest`, `ResetPartLayoutRequest`, `LayoutBlockResponse`, `PageResponse`,
  `ListPageOptions`.
- `BackgroundJobsContext.tsx:124` — `!job.id.startsWith("local-")` is the last live fragment of
  the loopback path; nothing has constructed a `local-` id since #060.
- `PageEditorProcessingBanner.tsx` is a `.tsx` with no JSX and no component.

**Config / deps**
- `protobuf` in the `inference` group — zero import sites; pulled in by `onnxruntime`, archived by
  ADR 0004. `security.yml:40-41` still explains the pin as live policy.
- `config/inference/trocr.yaml` — an engine the package does not have; the runner dispatches only
  `calamari` and `blla`.
- `.dockerignore:27` excludes `configs/`, removed by `4b1262f`.

**Tests**
- `_assert_adjacent_points_are_spaced` (`test_segment_refinement.py:77`) — defined, never called,
  so the minimum-vertex-spacing invariant *looks* covered and is not.
- Five unused settings imports in `tests/nomicous/integration/conftest.py:57-64`.
- `return_pooled_connections_before_leaving` and eight helpers copy-pasted across four device
  modules instead of living in `conftest.py`.

**Archive integrity.** `archive/onnx-runtime/` is half-archived: its tests and export scripts
import `inference.architectures.blla.onnx`, `preprocess_blla_image_numpy`,
`resize_heatmaps_nearest` and `src.model.inference_export.*` — all deleted by this merge. It
cannot be imported, let alone run. It also reaches *into* live code for a private name
(`_scaled_blla_width`). Nothing live imports it, so this is documentation debt, not a runtime bug.

---

## DOC AND BOARD DRIFT

- `issues/board.json:237,248,270` — 057 and 058 marked `ready`, 060 marked `backlog`, contradicting
  the same file's `stats` (14/14 done), its own kanban columns, and the issue files' frontmatter.
  Any regeneration that trusts `issues[].status` pulls three merged lanes back out of Done.
- `docs/merge-handoff.md` — obsolete, no superseded banner, and four separately-false
  instructions: merge a branch that no longer exists, a `packaging/helper/` path pinned by a test
  that no longer contains it, a test function that does not exist, a `--group export` that is not
  in `pyproject.toml`, and an `INFERENCE_SERVICE_SECRET` that a test now asserts is absent.
- `docs/merge-handoff-2026-08-04.md` — says of itself "disposable: delete it once both are
  merged". Both are merged. It also asserts `archive/onnx-runtime/` does not exist; it has 12
  tracked files.
- `todo.md:93-95` — "six signing secrets are still unset; the release workflow stays red by
  design." `release.yml:16-18` says "there are no signing secrets" (Trusted Publishing, OIDC), and
  `resume-inference-redesign.md` says eight existing secrets should be **revoked**. The todo would
  send someone to create the credentials #061 exists to eliminate.
- `docs/database-design.md:106,161` — the ER diagram still declares `INFERENCE_JOBS` with a full
  entity block, in a file whose prose carries the tombstone twice. The diagram is the part people
  copy from.
- `docs/guides/learnings.md:128-156` — two sections of current-tense operational guidance for the
  deleted loopback transport (helper CORS, Private Network Access preflight, the CSP loopback
  origins, `NEXT_PUBLIC_INFERENCE_HELPER_URL`). Indexed as "known platform pitfalls".
- `landing/index.html:185,239` — the **public marketing page** still sells the Inference Helper and
  weights that "sync from the registry". Highest-visibility surface in this audit and the only
  user-facing one.
- `docs/guides/testing.md:86` (port 8010), `nomicous/README.md:74` (port 8001),
  `deploy/platform/README.md:40` (links to a deleted README), `docs/README.md` (indexes none of
  the four handoff docs, including the one `issues/kanban.md` calls the first thing to read; points
  "deferred work" at the 26-line `docs/todo.md` rather than the 185-line root `todo.md`).
- Stale docstrings naming deleted subsystems: `inference/hub/artifacts.py:46` ("the helper
  capability document"), `:20` ("HTTP surfaces must map it to 503"), `weights/__init__.py:1`
  ("server-side cache layout"), `contracts/transcribe.py:36` ("this service"),
  `blla_runtime.py:41` ("sync run, queued job, helper"),
  `frontend/characterConfidence.ts:13` (`persistLocalTranscribe`, zero hits).

---

## WHAT IS GENUINELY GOOD

Not padding — these were probed specifically and held up:

- **Authorization is the strongest part of the merge.** `DocumentAccess` really does centralize
  what was thirty-odd repeated prefixes; no endpoint skips it; anonymous reads return 404-not-403
  for drafts so the public surface is not an existence oracle; the obvious IDOR in
  `_snapshot_or_404` is closed.
- **Device token handling is textbook.** 256-bit `token_urlsafe(32)`, stored only as a keyed
  HMAC-SHA256 digest, `hmac.compare_digest`, empty digest can never match, revocation re-read per
  request with no cache, raw token minted inside the verifying transaction and never persisted.
  Credential file: `0600` in a `0700` dir, `fchmod` after `os.open` to defeat umask, atomic
  temp-file rename.
- **The signed link itself is correct** (its redundancy is §5, not its cryptography): HMAC over
  key **and** expiry with a separator the key charset forbids, `compare_digest`, expiry inside the
  signed message, a strict key regex plus a filesystem containment check, and one indistinguishable
  403 for all three failure modes.
- **Two agents cannot claim the same page.** `with_for_update(skip_locked=True).limit(1)`, commit
  inside the lock, payload build deliberately outside the transaction. Sweeps re-assert their
  predicates on the `UPDATE` in the same transaction, so a row that changed hands is left alone.
- **Callback idempotency is real** — separate durable claim transaction, `FOR UPDATE` twice,
  terminal-status and already-claimed both return "already handled", merge and terminal write share
  one commit.
- **`test_migrations.py`** migrates a scratch database to head and asserts
  `compare_metadata(...) == []` with type and server-default comparison on. It would catch a
  missing migration.
- **Concurrency tests are not theatre** — real `ThreadPoolExecutor` over live HTTP and a `Barrier`
  forcing a simultaneous insert, asserting set-equality and exactly one `IntegrityError`.
- **`reset_supabase_nonprod.sh`** parses the env file directly rather than trusting the
  environment, captures the confirmation *before* the merge so an env file cannot confirm its own
  destruction, and rejects files mixing two project refs.
- **CI supply chain** — no `pull_request_target`, every action pinned to a full SHA, top-level
  `permissions: contents: read`, no `github.event.*` in any `run:`, Trusted Publishing with
  provenance attestation.
- **Calamari batch isolation genuinely isolates**; `torch.inference_mode()` everywhere, `eval()`
  asserted at both adapters, `weights_only=True` behind a SHA-256 artifact check, no `pickle`,
  `yaml.load`, `eval`, `exec`, or `shell=True` in scope; hub cache verifies digest and commit and
  `rmtree`s on every failure path.
- **Query keys are correct** — every key carries each entity id it reads, all primitives, real
  `enabled` guards, and `queryClient.clear()` on both set and clear of the access token so one
  user's reads cannot leak into the next session.
- **Optimistic rollback is consistent** — snapshot before write, restore on failure, with the
  read-modify-write-after-`await` hazard handled deliberately via refs and functional updaters.
- **Loopback removal was thorough at the transport layer** — every client, probe and CSP entry is
  gone; the one survivor is a filter (`local-` job ids), not a transport.
- **The code is honest about its accepted risks.** `device_service.py:32-38` and `devices.py:73-78`
  state the pairing-phishing gap outright rather than pretending the confirmation code closes it.
  That is why the nine comments in the table at the top matter so much: they are the exception to
  a codebase that otherwise tells the truth.

---

## SUGGESTED MERGE GATE

Block on: §1 (RCE), §2 (untested headline features), §3 (red CI), §4 (TLS), §9 (`file://`),
§10 (two DoS paths), §15 (reset crash), §18 (env template).

§5 is the one that deserves a design conversation rather than a patch — either the inline image
goes or the signed-link subsystem does, and whichever survives, ADR 0002's text has to match it.

Everything else is a follow-up issue.
