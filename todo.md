# TODO — remaining work after the 2026-08-03 review + implementation pass

Everything from the 109-finding review is implemented and unit-tested (backend 252 pass,
frontend 185 pass + typecheck clean, inference 159 pass / 1 skip / 1 fail). What follows is
what is **not** done. Ordered by what blocks what.

Current tree: committed on `feat/inference-cli-redesign`. The 2026-08-04 inference
redesign (ADRs 0002-0005, issues #48-#66) is layered on top; see
`docs/merge-handoff-inference-redesign.md` before merging anything into it.

---

## P0 — Blocks shipping

### 1. ~~Bump the pinned `blla.onnx` digest~~ - dissolved by ADR 0004 / issue 049

This was a four-step ordered dance: upload the re-exported `blla.onnx` to the Hub
*first* (because `src/hf/resolve/cache.py` rmtree's the cache on manifest mismatch
and `src/hf/cache/` is untracked), then bump `hub_revision` and `artifact_sha256`,
then the hardcoded copies in `test_registry.py`, then the local cache copy. It was
caused by the GroupNorm export fix changing the ONNX graph, so the artifact hash
moved away from what the registry pinned.

Retiring the ONNX runtime removed the problem rather than solving it. The registry
now pins `blla.safetensors` (`8b5b6ec2...`) and `best.pt` (`ea711b91...`) at the
same already-published **Hub revision**, both verified against the live Hub. There
is no re-export step, so there is no artifact that can drift from the checkpoint it
came from.

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

### 4. Benchmark ONNX vs Torch **closure** sizes, then decide

ADR 0004 retired ONNX on measurements that were real but not comparable, and the
one number that matters was never taken.

What we measured:

| | measured |
|---|---|
| `onnxruntime` package, installed | 63 MB |
| `torch` package, installed | 388 MB |
| **Full `nomicous-inference` closure, CPU-pinned (Linux)** | **969 MB** |
| Full closure, no pin (Linux) — 16 CUDA wheels | **4801 MB** |
| Full closure, macOS arm64 | 811 MB |

The first two rows compare *packages*. The last three measure a *closure* — Torch
plus OpenCV, SciPy, NumPy, SymPy, scikit-image and the rest of what a researcher
actually downloads. **There is no ONNX counterpart to those three rows**, so the
comparison that would justify or overturn the decision has never been run: nobody
has built the equivalent ONNX-runtime closure and weighed it against 969 MB.

That gap matters more now than when ADR 0004 was written, because the closure came
in far worse than the ADR assumed — 2.5x its stated figure at best, 12x at worst,
and the worst case is what a plain `pip install` produces (issue #65).

Take both closures on the same machine, same lockfile discipline, same platforms
(Linux x86_64/aarch64, macOS arm64, Windows), and report installed size, bytes
fetched, and cold-cache wall clock. Include peak RSS while segmenting one page —
issue #62 measured 3.0 GB for ONNX against 7.0 GB for Torch, which is the same
question in memory rather than disk, and on an 8 GB laptop it may matter more.

Then decide, on the numbers:

- **Keep Torch** if the closure gap is modest. Output is byte-identical, PyTorch is
  14%/40%/12% faster (ADR 0004), and the ONNX↔Torch parity apparatus stays deleted
  — that was the recurring cost the owner named.
- **Revisit** if the gap is large. Reversal is now cheap by design: the ONNX
  implementation was archived rather than deleted, under `archive/onnx-runtime/`
  with a README and a revival checklist, precisely so this stays a live option.

Do not decide this from the package-level 63 MB vs 388 MB figures. They are the
numbers that made the original argument and they are the wrong unit.

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
- The graph-shape guard inspects the committed blob rather than a freshly exported one.

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
