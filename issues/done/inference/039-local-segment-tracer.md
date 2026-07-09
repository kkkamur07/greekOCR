---
id: "039"
title: "local-segment-tracer"
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

Extend the local inference path to **Auto-segment**: when the **Inference helper** is healthy, the segment model has `host_eligibility: local`, and **Inference preference** is not cloud-forced, the browser calls `POST localhost:8001/inference/v1/run` with task `segment` (including existing Otsu refinement params), then persists merged layout through the hosted API.

Add an authenticated persist endpoint that applies **Segment merge** server-side — update/add/prune machine lines, respect manual geometry, prune transcriptions on removed machine lines — matching cloud segment job callback behavior.

Frontend: branch `runAutoSegment` in the page editor the same way pairing assist branches for transcribe.

Mark `greek-kraken-segment-v1` `host_eligibility: local` in **Registry** (if not already done in 038).

## Acceptance criteria

- [x] Helper `/run` segment returns canonical segment output (blocks, lines) without service secret.
- [x] Authenticated persist endpoint applies **Segment merge** rules identical to cloud segment job completion for the same input.
- [x] Manual geometry lines are not overwritten by local segment persist.
- [x] Auto-segment uses local path when helper up + model `local` + cloud toggle off; otherwise enqueues **Product job** + **remote inference**.
- [x] Otsu refinement options from the editor are forwarded to local `/run` and affect persisted layout.
- [x] Integration tests cover local segment persist merge behavior against a fixture layout.

## Blocked by

- [038-inference-helper-local-transcribe-tracer.md](038-inference-helper-local-transcribe-tracer.md)

## User stories covered

- 19, 28, 29, 30, 31, 33, 34, 46, 48, 49
