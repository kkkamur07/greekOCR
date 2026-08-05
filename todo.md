# TODO — remaining work after the 2026-08-03 review + implementation pass

Everything from the 109-finding review is implemented and unit-tested (backend 252 pass,
frontend 185 pass + typecheck clean, inference 159 pass / 1 skip / 1 fail). What follows is
what is **not** done. Ordered by what blocks what.

Current tree: 159 files changed, uncommitted on `main`. No `git add`/`commit`/`push` has been
run — the whole pass is still in the working tree.

---

## P0 — Blocks shipping

### 1. Bump the pinned `blla.onnx` digest (**one failing test depends on this**)

The GroupNorm export fix changed the ONNX graph, so the artifact hash moved. The registry
still pins the old one, and the resolver correctly refuses to load the new file.

- old: `5871e3755d414c00380794bafd570c1bb3d6a3255cdfb11b1bbe99dcec084d5e` (5,077,600 B)
- new: `d3e9c086541157a2f55209bc4802206478231e7637c12ee4884504f94d6c4ed3` (5,102,555 B)

Four things change together, and **the order matters**:

1. **Upload to `hf://kkkamur07/segmentation-blla@stable` FIRST.**
   `src/hf/resolve/cache.py` runs `shutil.rmtree(cache_dir)` on manifest mismatch — and again
   in its `except` handler. `src/hf/cache/` is untracked, so git cannot recover the weights.
   Bumping the pin before the Hub serves the new blob destroys the local cache with no undo.
2. `inference/registry.yaml:35-36` — `hub_revision` **and** `artifact_sha256`.
3. `tests/inference/unit/test_registry.py:38,40` — both values are hardcoded there too.
4. `src/hf/cache/blla-segment/stable/blla.onnx` + `.hub-manifest.json` — refresh the copy.

Failing test that goes green afterwards:
`tests/inference/unit/test_blla.py::test_standalone_helper_returns_onnx_blla_response_for_real_image`
(currently 503). It was left red on purpose — making it pass locally would hide an incomplete
shipping step.

### 2. Set `DEVICE_TOKEN_HMAC_SECRET` before rotating `JWT_SECRET`

`JWT_SECRET` now requires ≥32 bytes and ≥128 bits of entropy in production. **The API refuses
to boot if the live value fails the check** — so verify the deployed value before deploying.

`DEVICE_TOKEN_HMAC_SECRET` falls back to `JWT_SECRET` when unset. Set it explicitly *first*,
otherwise rotating `JWT_SECRET` silently unpairs every helper device.

### 3. Confirm the `platform-worker` host is actually deployed

`claim_next_pending_job` has exactly one caller: `python -m backend.jobs.worker_main`. Vercel
runs with `JOB_WORKER_ENABLED=false` (it is request/response only), so if that separate host
isn't running, **nothing claims `pending` jobs in production**. The new on-read stale sweep
covers timeouts but does not claim work.

### 4. Commit

14 workstreams sitting uncommitted on `main`. Worth splitting into logical commits and keeping
the pre-existing modifications (the ones present before this pass) separate.

### 5. Release signing secrets

Six secrets are still unset; the release workflow stays red **by design** until they exist.

---

## P1 — Correctness / safety

### 6. `bounded_image` is not thread-safe

Introduced by this pass's own `asyncio.to_thread` work. Shared mutable state now reachable from
multiple worker threads.

### 7. BLLA residual divergence at extreme widths

The fix bounds the accumulator; it does not remove the mechanism.

| scaled width | max logit delta | sigmoid flips |
|---|---|---|
| 2400–2700 (real pages) | 1.8e-04 | 0 |
| 6000 | 0.623 | 2 |
| 12000 | — | 3 |
| 14400 | — | 13 |

`MAX_WIDTH_TO_HEIGHT_RATIO = 8` permits up to 14400. Either lower the clamp or add a third
reduction stage to `_ExportGroupNorm`.

### 8. BLLA regression guards have three blind spots

- They read only the staging path, so a stale `src/hf/cache/` copy is invisible to them.
- The parity suite bypasses the resolver entirely.
- The graph-shape guard inspects the committed blob rather than a freshly exported one.

### 9. Pin the helper download URLs

Currently unversioned. Pin to a specific release and verify against `SHA256SUMS`.

### 10. conftest environment pollution

`tests/nomicous/integration/conftest.py` sets `INFERENCE_DATABASE_URL` via `setdefault` at
import time, which makes one security test vacuous — it asserts against a value the test file
itself supplied.

**Standing hazard, unrelated to the bug above:** `_truncate_database()` is `autouse=True` and
issues `TRUNCATE TABLE <every table> RESTART IDENTITY CASCADE` before **every** test. The only
thing standing between that and production is the `os.environ.setdefault("SYNC_DATABASE_URL",
localhost:5433)` on line 22. Never export a Supabase `SYNC_DATABASE_URL` before running this
suite.

Related: `nomicous/backend/core/.env` is absent, so settings resolve to `.env.supabase` → the
live pooler. Worth adding a local `.env` so nothing defaults to production.

---

## P2 — Deferred / cleanup

### 11. Integration suite unverified

Deferred by request. Needs a throwaway Postgres (local or a scratch Supabase project) before it
can run — see the hazard in #10.

### 12. 519 ruff violations behind a per-file-ignores allowlist

`src/model` holds 385 of them, including real `F901` / `F403` / `F841`.

### 13. Build the `/pair` consent page + device management UI

Backend is complete and tested; the feature flag is off. Frontend is the remaining half.

### 14. Regenerate `openapi.json` and `schema.d.ts`

Stale after the device-pairing routers and the job-lifecycle schema changes.

### 15. Docs: stale `/inference/v1/catalog` references

That route was removed and folded into `/inference/v1/info`.

---

## Reference

- Full review: `nomicous-deep-review.md` in the session scratchpad (2,004 lines, 109 findings)
- BLLA root cause: `nn.GroupNorm` lowers to `Reshape([0,32,-1]) → InstanceNormalization`,
  flattening a group to 2,224,800 floats on a 2471px page. ORT's CPU kernel reduces that
  serially in one float32 accumulator (torch uses blocked/Welford), giving ~1000× worse error
  on spatially-correlated post-ReLU activations. 12–24 pixels crossed the 0.5 sigmoid boundary;
  the decoder is discontinuous there, so short lines restructured entirely → IoU 0.5026.
  Fixed by a trace-only staged reduction. Now 0/518 lines below threshold, min IoU 1.0000.
