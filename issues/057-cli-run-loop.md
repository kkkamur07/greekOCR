---
id: "057"
title: "cli-run-loop"
type: AFK
status: done
tracker: "https://github.com/kkkamur07/greekOCR/issues/57"
blocked_by:
  - "053-signed-page-image-link.md 054-device-lease-stale-sweep.md 056-cli-pair-and-version.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

The run loop — the thing that actually makes a researcher's laptop useful. Claim a page, fetch its image over the signed link, run the model, post the result back, repeat.

This closes the four-step path ADR 0003 is built around: enqueue, claim, run, callback. One database, one HTTP hop, no inbound connection, no open port.

Per-page progress is shown in the terminal, because the researcher needs to see it is working. Ctrl-C finishes or explicitly fails the page in flight before exiting, so a considerate shutdown never leaves a job stuck — and a crash is still covered by the lease.

A page that fails on this machine is reported as failed with its reason. The researcher is never left waiting on a page that already died.

A flag chooses between exiting when the queue is empty and waiting for more, so the same binary serves an interactive session and a script.

The hosted worker runs this same loop with a service credential and a short poll.

## Acceptance criteria

- [x] Claim, fetch, run, and callback complete end to end against a real platform and real ~~ONNX~~ **Hub artifact**s
- [x] Exactly one page is in flight at a time
- [x] Per-page progress is visible in the terminal
- [x] Ctrl-C reports the in-flight page terminally before exiting
- [x] A page that fails to run is reported failed with its reason, and the loop continues
- [x] A killed process leaves a page that the lease later releases
- [x] Empty queue is handled by both exit-when-empty and keep-waiting behaviour
- [x] Output from the CLI matches output from the same code run in-process, on the same page
- [x] Tested by running the real CLI process against a real running platform

## As built

**"Real ONNX artifacts" above is stale.** [ADR
0004](../docs/adr/0004-pytorch-is-the-inference-runtime.md) reversed it between
this issue being written and being picked up: PyTorch is the inference runtime,
ONNX is archived under `archive/onnx-runtime/`, and the trained artifact and the
run artifact are now the same file. The end-to-end tests run the real PyTorch
**Hub artifact**s through the **Hub cache**, which is what the criterion was
asking for.

- `inference/cli/run.py` - the loop. `inference/cli/api.py` gained `claim_page`,
  `fetch_page_image`, and `report_page` on the existing `PlatformClient`.
- `--exit-when-empty` is the flag; without it the loop waits. `--wait-seconds`
  overrides the per-claim long poll, which defaults to 25 for a paired machine
  and 0 for a hosted worker.
- A hosted worker is the same loop with `NOMICOUS_SERVICE_TOKEN` set. The
  credential is read from the environment rather than a flag, because a token on
  a command line is a token in `ps` output.
- The model runtime is imported inside the call that needs it, so `nomicous
  version` still does not pay for Torch.
- Tests: `tests/inference/integration/test_cli_run.py`, eleven of them, all
  live - real wheel with its real closure, real uvicorn on real Postgres, real
  weights, real signals. Three platform processes, because the **version floor**,
  the short **lease**, and the ordinary configuration are settings one process
  cannot hold two values of.

Nothing was deferred, and nothing here is untested.

### Found while building, not fixed here

`ClaimedPageResponse.request` is a whole `JobSubmitRequest`, and
`build_inference_submit_request` populates its `image_bytes` from the media
store - so **the claim response carries the page image twice**: inline as base64
*and* as the `page_image_url` this issue fetches through. ADR 0002 rejected
streaming scans through the API because "the production API is serverless, so
streaming manuscript scans through it costs money for nothing", and the inline
copy does exactly that on every claim. The run loop uses the signed link and
ignores the inline bytes, so this is a platform-side cost rather than a defect in
the agent. Removing it means changing the claim contract and the assertion at
`tests/nomicous/integration/test_device_claim.py:182`, which belongs to #53
rather than here.

## Blocked by

- #53
- #54
- #56
