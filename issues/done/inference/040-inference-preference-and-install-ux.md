---
id: "040"
title: "inference-preference-and-install-ux"
type: AFK
status: done
blocked_by:
  - "038-inference-helper-local-transcribe-tracer.md"
parent_prd: "issues/done/inference/prd-local-inference-helper.md"
triage: ready-for-agent
---

## Parent

`issues/done/inference/prd-local-inference-helper.md` — Local inference via **Inference helper** (browser-orchestrated).

## What to build

Polish researcher-facing UX for **Inference helper** discovery and **Inference preference** beyond the minimal toggle in 038.

When health probe fails, show a non-blocking banner explaining local inference benefits with a link to install the **Inference helper** (placeholder URL or docs page until 041 ships installers). On shared machines with no helper, cloud path works silently — no error modal.

During first-time model download on the helper, show progress copy ("Downloading {registry model id}…") with **Use cloud instead** that cancels local wait and routes the current action to **remote inference**.

Persist **Inference preference** across sessions (`localStorage` minimum; optional user settings API if already present). Settings UI copy: **Use cloud inference** with short explanation that local is default when helper is installed.

When selected model has `host_eligibility: remote`, hide or disable local option and explain that the model runs on server only.

## Acceptance criteria

- [x] Helper absent: banner with install CTA; pairing assist and auto-segment still work via cloud without blocking errors.
- [x] Model downloading locally: visible progress state and **Use cloud instead** switches current action to **Product job** path.
- [x] **Inference preference** survives page reload.
- [x] `host_eligibility: remote` models never attempt localhost `/run`; UI communicates server-only models.
- [x] Frontend tests cover probe-fail → cloud routing and cloud-toggle persistence (mocked `fetch`).

## Blocked by

- [038-inference-helper-local-transcribe-tracer.md](038-inference-helper-local-transcribe-tracer.md)

## User stories covered

- 7, 13, 14, 20, 22, 32, 37, 39, 42
