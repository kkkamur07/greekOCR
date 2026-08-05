---
id: "057"
title: "cli-run-loop"
type: AFK
status: backlog
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

- [ ] Claim, fetch, run, and callback complete end to end against a real platform and real ONNX artifacts
- [ ] Exactly one page is in flight at a time
- [ ] Per-page progress is visible in the terminal
- [ ] Ctrl-C reports the in-flight page terminally before exiting
- [ ] A page that fails to run is reported failed with its reason, and the loop continues
- [ ] A killed process leaves a page that the lease later releases
- [ ] Empty queue is handled by both exit-when-empty and keep-waiting behaviour
- [ ] Output from the CLI matches output from the same code run in-process, on the same page
- [ ] Tested by running the real CLI process against a real running platform

## Blocked by

- #53
- #54
- #56
