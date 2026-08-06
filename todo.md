# TODO — after the 2026-08-06 validate-and-build sweep

The previous list is done, and a third of it turned out not to need doing. Every entry
below was checked against the tree before it was worked on; where the old entry was wrong,
the correction is kept rather than deleted, so the same wrong diagnosis is not rediscovered.

Branch: `feat/todo-sweep` (three commits, off `main` at `6c0bb7b`). Nothing is pushed.

**Verified green on this branch**, each run to completion:

| suite | result |
|---|---|
| `pytest tests/nomicous` (incl. integration) | 746 passed, 1 skipped |
| `pytest tests/export tests/inference tests/hf` | 316 passed, 7 skipped |
| frontend `vitest` | 51 files, 252 passed |
| `tsc --noEmit` / `eslint .` | clean / 0 errors, 2 pre-existing warnings |
| `ruff check . src/model/inference_export` | All checks passed |
| `npm run check:api` | exit 0 |

The 7 skips are the published-artifact assertions, which need `src/hf/cache/`. That
directory does not exist in a fresh worktree. **Run them in the main checkout before
trusting anything about the published `blla.onnx`** — a half-landed exporter change would
be silently green here.

---

## A — Needs your decision. No agent should take these.

### A1. Secrets are in the public git history

`infrastructure/.env` was committed in **`e81a50c`** (2026-05-21) with `DATABASE_URL`,
`SYNC_DATABASE_URL` and `JWT_SECRET`. It is reachable from `origin/main`, and the
repository is public. The file was deleted with the rest of that tree in `784df29`, but the
blob stays retrievable forever.

The severity is lower than it first looks, and this was verified rather than assumed:
`ENVIRONMENT=development`, both database URLs are `localhost:5433`, `CORS_ORIGINS` is
localhost-only, and the `JWT_SECRET` is **23 characters** — below the 32-byte floor
`AuthSettings._validate_secret` enforces, so it could not boot a current production API.
This is dev scaffolding, not a live breach. `.gitignore:15` covers the path now.

Three decisions, all yours:

1. **Rotate that `JWT_SECRET` and database password only if either was ever reused** in a
   reachable environment. No evidence of reuse was found, but only you can confirm that.
2. **History rewrite** (`git filter-repo`/BFG + force-push) is destructive to every
   collaborator and to the three worktrees currently sharing this checkout. Not done.
3. A gitleaks rule for this class of secret **is** now in place (see the security commit),
   so the gap that let it through is closed going forward.

### A2. A sub-agent wrote live `.env` values into a transcript on disk

During the security audit, a sub-agent attempted to send real local secrets — JWT and
inference secrets, a Supabase service-role key fragment — in plaintext to another agent.
**The send failed and nothing left the machine.** But the plaintext now also sits in
`~/.claude/projects/.../subagents/agent-ac26f0cc0ce0e9718.jsonl`, in addition to the
gitignored `.env` files it was read from.

Purge that transcript if you want it gone. Local-only, already-local secrets, no
exfiltration — but it is a second copy you did not ask for.

### A3. `torch>=2.13.0` would close CVE-2025-3000, and is coupled to A4

`train` and `export` both floor at `torch>=2.10.0`. `PYSEC-2025-194` / `CVE-2025-3000` is
fixed in 2.13.0, so this is a "not yet upgraded" gap rather than a "no fix exists" one —
`docs/security/vex-torch-pysec-2026-139-cve-2025-3000.md` now says so plainly.

Not done because torch **builds the exported ONNX graph**. A bump can change the artifact,
which means re-export, re-upload to the Hub, and re-pinning digests in four places. That is
the same re-publish A4 would need. **If you do one, do both — it costs one re-publish
instead of two.** The runbook is in A4.

`PYSEC-2026-139` / `CVE-2026-4538` has no fix at any version. That ignore stays regardless.

### A4. The BLLA clamp went 8 → 3, and it costs resolution

This is done and shipped on the branch, but it is a **product** trade-off, and reversing it
is one line in `inference/architectures/blla/blla_preprocessing.py`.

Measured on the real `segment_page.jpeg` fixture tiled to each width, shipped ONNX graph
against the Torch oracle:

| scaled width | rms &#124;Δ&#124; | max &#124;Δ&#124; | logits crossing 0.5 |
|---|---|---|---|
| 2471 (a real page) | 1.74e-05 | 1.55e-03 | 0 |
| 3600 | 2.19e-05 | 1.10e-03 | 0 |
| 5400 (**the new bound**) | 3.59e-05 | 2.34e-03 | 0 |
| 7200 | 7.47e-05 | 1.24e-02 | 1 |
| 9000 | 9.15e-05 | 2.15e-02 | 1 |
| 14400 (**the old bound**) | 1.95e-04 | 2.38e-02 | 3 |

**The cost:** any source wider than 3:1 is now squeezed horizontally by up to 2.67× where
8:1 passed before. A codex leaf (~0.7:1) and a two-page spread (~2.5:1) are untouched; a
stitched scroll pays. Weighed against 3 threshold crossings out of 6.48M logits at the old
bound — and ADR 0006's incident involved 12–24 — **4 (7200px) is a defensible middle**
if you value panorama resolution more than I did. Say the word and it is a one-line change.

### A5. The Safari `/auth/refresh` 403 fix is a candidate, not a confirmation

Shipped on the branch and **cannot be verified without a real Safari** against the deployed
hosts. It is additive and falls back to the old cookie read, so it should not regress
Chrome or Arc — an in-memory-vs-cookie staleness regression across two tabs was found
during the work and closed with a one-shot retry.

It also **removes the double-submit cookie check** from `_require_csrf`, keeping only the
synchronizer-token hash. The reasoning: double submit is a substitute for a server-side
per-session secret, not a layer on top of one, and no attacker is stopped by the equality
check that is not already stopped by the hash. `test_a_csrf_cookie_alone_still_authorises_nothing`
presents another session's token in both cookie and header and still gets 403. Sound — but
it is a deliberate reduction of defence-in-depth on the auth path, made against an
**unconfirmed** diagnosis. Your call whether it deploys.

To confirm in Safari Web Inspector, signed in, against the deployed hosts:

1. **Storage → Cookies** on `app.nomicous.com` *and* `api.nomicous.com`. Is `greekocr-csrf`
   present under each, and do the values match? That distinguishes blocked / partitioned / fine.
2. **Console** on `app.`: does `document.cookie` show `greekocr-csrf`? That is the exact
   read the old code depended on.
3. **Network → `POST /auth/refresh`.** Is `X-CSRF-Token` present, and does it match the
   `csrf_token` in the preceding `/auth/login` response body? Is `__Host-greekocr-session`
   attached? 200 or 403?
4. Repeat past the 15-minute access-token expiry, and with **two tabs open**.

If you see **401 rather than 403**, Safari is dropping the *session* cookie and this change
is irrelevant to the cause.

### A6. Owner actions carried forward, unchanged

- **Revoke eight CI secrets.** All eight are confirmed unreferenced by any workflow — safe
  to revoke. List and the credentials to revoke alongside them: §8 of
  [`docs/resume-inference-redesign.md`](docs/resume-inference-redesign.md).
- **Set `DEVICE_TOKEN_HMAC_SECRET` before rotating `JWT_SECRET`.** The *code* half of the
  old entry is already shipped — `DeviceSettings._validate_production_credential_key()`
  refuses to boot in production when the secret is unset or equal to `JWT_SECRET` and
  pairing is enabled. What remains is setting a real distinct value in the live environment.
- **Confirm the `platform-worker` host is actually deployed.** Now observable rather than
  silent: `/health` reports `oldest_pending_job_seconds` and warns past
  `JOB_QUEUE_STALL_WARNING_SECONDS` (900s). There is still no IaC for that host anywhere in
  the repository — standing it up is a manual step in `docs/deployment/production.md` §3.
- **The push itself.** This branch has never been pushed, and neither has the ~100-commit
  inference redesign under it. §7 of the resume doc: no agent decides that.

---

## B — Real work, not yet done

### B1. `src/` holds 576 ruff findings and is excluded wholesale

`pyproject.toml` `extend-exclude` lists `src`, which is a suppression and says so in its own
comment. `src/model` alone has 380, including `F901`, `F403` and `F841` — the only
genuine-bug-class violations left anywhere in the repository.

Not touched because `src/` is audit-only in this repo by standing instruction: vendored
Calamari and the research trees are not maintained to a lint standard. `src/model/inference_export/`
is the exception, is ours, and is **already clean**.

**This needs your approval to proceed**, and it is not a small job. If you want it, the
sensible order is `F` rules only first (real bugs), leaving `I`/`UP`/`SIM` alone.

### B2. The click VEX cannot retire yet

`uv.lock:351` resolves Click **8.2.1**, below the 8.3.3 floor. When a future `uvicorn` or
`typer` bump clears it: delete `docs/security/vex-click-pysec-2026-2132.md` and the
`--ignore-vuln PYSEC-2026-2132` line in `.github/workflows/security.yml` together.

### B3. mypy runs with `continue-on-error: true`

`.github/workflows/quality.yml:69`. A deliberate, documented ratchet against an untyped
codebase — the comment says to drop it once the count reaches zero. Named here so it is a
choice rather than something nobody looks at. Not a security gate; every security gate in
CI was audited and none of them can pass while failing.

### B4. Smaller things found along the way, none urgent

- `devicesApi` sits in `src/api/resources.ts` because `client.ts` was owned by another lane
  at the time. It belongs in `client.ts` with every other API call.
- `.pub-pdf-view__actions` uses inline styles; `theme-shell.css` is its proper home.
- `docs/deployment/production.md` §3 could now point at `oldest_pending_job_seconds` as the
  way to tell whether the manual worker step was actually done.
- `JOB_QUEUE_STALL_WARNING_SECONDS` is not in `nomicous/backend/core/.env.example` —
  consistent with the other sweep knobs, which are also absent.
- A full-history `gitleaks detect` has never completed; two attempts ran ~45–50 minutes
  against large historical blobs and were killed. The A1 finding was located by pickaxe
  search instead. Worth one authoritative run on an idle machine.
- Adding a local `nomicous/backend/core/.env` would stop settings falling through to
  `.env.supabase` (the live pooler) by default. The truncate guard now makes the dangerous
  half of this safe, but the fallback itself is still surprising.

### B5. Grep-style assertions that could execute instead

Carried forward from `docs/test-hardening-handoff.md`, deleted once `test/suite-hardening`
merged. **Partly overtaken**: the suite-reduction pass deleted a number of these greps
outright rather than converting them, on the grounds that an integration test already
proved the property. Re-check each against the tree before working it — the handoff's line
numbers are dead, and `test_device_pairing.py` and `test_job_lifecycle.py` were both
rewritten since.

What it proposed making executable, where the property is still only asserted by substring
search: reset-script guards via a stubbed `psql`; role grants via `pg_roles` /
`role_table_grants`; the advisory try-lock across two sessions; an `asyncio.to_thread`
thread-id spy; `build.sh DEST="/"`; font-resolver equality and Greek `_render_pdf`;
pip-uninstalled as a CI step. Migration columns are **done** —
`integration/test_migrations.py` diffs the migrated schema against `Base.metadata`.

Also unresolved from that branch: one full-lane run showed
`test_device_lease.py::test_a_platform_dispatched_page_still_fails_on_the_waiting_timeout`
red, and re-running that file passed 17/17 twice. Suspected ordering/loop interaction,
never root-caused. Related to the known asyncpg "attached to a different loop" issue (#63).

---

## C — Closed by this sweep, kept so it is not re-litigated

- **The combined inference suite runs green.** `pytest tests/inference tests/export tests/hf`
  → **321 passed, 0 failed** in 367s on an idle machine, in the main checkout. This was the
  one P0 claim about ADR 0006 never verified as a whole. It is verified.
- **The integration suite is no longer unverified.** 746 passed, 1 skipped.
- **`bounded_image` was never unsafe.** It reads `image.size` from the header and never
  mutates `Image.MAX_IMAGE_PIXELS`; `test_concurrent_decodes_do_not_interfere` drives 64
  decodes across 8 threads and already proved it. The entry was written defensively in the
  same commit that fixed it.
- **`openapi.json` and `schema.d.ts` were never stale.** They regenerate byte-identical, and
  `quality.yml:154-175` has been drift-checking them on every PR the whole time.
- **BLLA item 8's diagnosis was wrong twice.** Its table is the *pre-fix unstaged* graph —
  disabling the staged reduction reproduces 0.574 at width 6000 against the tabulated 0.623,
  while the shipped graph measures 2.9e-03 there with zero crossings, ~200× better. And the
  third reduction stage it proposed is **measurably useless**: accumulating `Gn_13`'s moments
  in float64 reproduces the float32 staged error to six figures, so the residual is Torch's
  own float32 `group_norm`, not ORT's serial reduction. No exporter change reaches it.
- **The PDF preview bug was not Safari's.** `vercel.json` ships `object-src 'none'`, so the
  `<object>` embed was blank in every conforming browser; it only looked browser-specific
  because the dev server does not serve that header. Fixed with `<iframe>` +
  `frame-src 'self' blob:`. `blob:` is **not** covered by `'self'` — switching to an iframe
  alone would not have worked.
- **The 519-ruff-violations entry conflated two mechanisms.** `src/` is excluded wholesale,
  not held behind `per-file-ignores`. The editable trees held 186, all now fixed. Deleting
  the blanket `I001` exposed why it existed: isort's `src` named packages instead of the
  roots containing them, so first-party imports sorted as third-party.
- **The device-secret code fix already shipped.** `DeviceSettings._validate_production_credential_key()`
  has been refusing the silent fallback in production for a while.

---

## Reference

- BLLA root cause, still accurate: `nn.GroupNorm` lowers to `Reshape([0,32,-1]) →
  InstanceNormalization`. The staged reduction in `src/model/inference_export/blla/export.py`
  fixed the catastrophic case (IoU 0.5026 → 1.0000). What remains is width-proportional and
  lives in Torch's own float32 accumulation, not in the export.
- `docs/resume-inference-redesign.md` — where the inference redesign stands, §5's traps, and
  the owner actions in §7–8.
- `docs/merge-handoff-inference-redesign.md` — history; open it only to find out *why* a
  conflict was resolved the way it was.
