# TODO — remaining work after the 2026-08-03 review + implementation pass

Everything from the 109-finding review is implemented and unit-tested (backend 252 pass,
frontend 185 pass + typecheck clean, inference 159 pass / 1 skip / 1 fail). What follows is
what is **not** done. Ordered by what blocks what.

Current tree: committed on `feat/inference-cli-redesign`. The 2026-08-04 inference
redesign (ADRs 0002-0005, issues #48-#66) is layered on top; see
`docs/resume-inference-redesign.md` before merging anything into it.

---

## P0 — Blocks shipping

### 1. ~~Bump the pinned `blla.onnx` digest~~ - done 2026-08-05 (ADR 0006)

ADR 0004 dissolved this by retiring the ONNX runtime rather than by doing it, and
that turned out to matter: the `blla.onnx` published at `444d51dd` was the
**pre-fix** export all along, carrying six `InstanceNormalization` nodes. So the
GroupNorm fix existed in the exporter and had never reached the artifact.

ADR 0006 put the ONNX runtime back, which made this blocking again, and the dance
was run in the order this entry always specified: re-export, upload to the Hub
first, then `hub_revision` + `artifact_sha256`, then the hardcoded copies in
`test_registry.py`, then the local `src/hf/cache/` copy.

The registry now pins `blla.onnx` (`d3e9c086...`) at revision `5c20a584...` and
`best.onnx` (`3cb01b58...`) at the unchanged `5ff715e8...`. Calamari needed no
re-publish: re-exporting `best.pt` today reproduces the published graph byte for
byte.

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

### 4. ~~Benchmark ONNX vs Torch **closure** sizes, then decide~~ - done 2026-08-05

Measured on the same machine, same method as the 811 MB row, macOS arm64:

| closure | installed |
|---|---|
| Torch (ADR 0004) | **817 MB** |
| ONNX Runtime (ADR 0006) | **372 MB** |

The gap is 445 MB, 54% of the install, and `onnxruntime` (70 MB) replaces `torch`
(475 MB) plus the SymPy/networkx/torchgen tail that came with it. That is the
"revisit" branch this entry described, and it was taken: see
[ADR 0006](docs/adr/0006-onnx-runtime-is-the-inference-runtime.md), which also
records the two things the disk figure understates - the CPU pin that could not
be expressed in package metadata (4801 MB on a plain Linux `pip install`), and
the 3.0 GB vs 7.0 GB peak RSS from issue #62.

The latency the decision gives up is real and was weighed: PyTorch is 40% faster
on the forward pass, which is 12-14% of a page end to end, because the NumPy and
scikit-image decoder dominates and is unchanged either way.

### 5. ~~Commit~~ - done

The pass is committed on `feat/inference-cli-redesign`, split into logical commits.
The inference redesign (issues #48-#61) landed on top of it in per-issue lanes.

### 6. Release signing secrets

Six secrets are still unset; the release workflow stays red **by design** until they exist.

---

## P1 — Correctness / safety

### 7. `bounded_image` is not thread-safe

Introduced by this pass's own `asyncio.to_thread` work. Shared mutable state now reachable from
multiple worker threads.

### 8. BLLA residual divergence at extreme widths

**Live again as of ADR 0006** - this was moot only while PyTorch was the runtime.
The fix bounds the accumulator; it does not remove the mechanism.

| scaled width | max logit delta | sigmoid flips |
|---|---|---|
| 2400–2700 (real pages) | 1.8e-04 | 0 |
| 6000 | 0.623 | 2 |
| 12000 | — | 3 |
| 14400 | — | 13 |

`MAX_WIDTH_TO_HEIGHT_RATIO = 8` permits up to 14400. Either lower the clamp or add a third
reduction stage to `_ExportGroupNorm`.

### 9. BLLA regression guards have three blind spots

- They read only the staging path, so a stale `src/hf/cache/` copy is invisible to them.
- The parity suite bypasses the resolver entirely.
- ~~The graph-shape guard inspects the committed blob rather than a freshly exported one.~~
  Closed by ADR 0006, and closed by asserting *both*: one test exports and checks the
  result, a second checks the **published** artifact. Splitting them is what surfaced the
  fact that the exporter was correct while the published file was not.

### 10. ~~Pin the helper download URLs~~ - dissolved by issue 061

There are no release assets and no `SHA256SUMS` manifest to verify against: the
per-OS installers and their signing pipelines are deleted and the distribution
is PyPI. The frontend constants that still build `releases/latest/download/…`
URLs go with the loopback path in #60.

### 11. conftest environment pollution

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

### 12. Integration suite unverified

Deferred by request. Needs a throwaway Postgres (local or a scratch Supabase project) before it
can run — see the hazard in #10.

### 13. 519 ruff violations behind a per-file-ignores allowlist

`src/model` holds 385 of them, including real `F901` / `F403` / `F841`.

### 14. Build the `/pair` consent page + device management UI

Backend is complete and tested; the feature flag is off. Frontend is the remaining half.

### 15. Regenerate `openapi.json` and `schema.d.ts`

Stale after the device-pairing routers and the job-lifecycle schema changes.

### 16. ~~Docs: stale `/inference/v1/catalog` references~~ - done in issue 061

Corrected to `/inference/v1/info` in the root README, the hosting guide, and the
model checklist. `tests/inference/unit/test_helper_app.py` asserts the old route
returns 404.

---

## Reference

- Full review: `nomicous-deep-review.md` in the session scratchpad (2,004 lines, 109 findings)
- BLLA root cause: `nn.GroupNorm` lowers to `Reshape([0,32,-1]) → InstanceNormalization`,
  flattening a group to 2,224,800 floats on a 2471px page. ORT's CPU kernel reduces that
  serially in one float32 accumulator (torch uses blocked/Welford), giving ~1000× worse error
  on spatially-correlated post-ReLU activations. 12–24 pixels crossed the 0.5 sigmoid boundary;
  the decoder is discontinuous there, so short lines restructured entirely → IoU 0.5026.
  Fixed by a trace-only staged reduction. Now 0/518 lines below threshold, min IoU 1.0000.
