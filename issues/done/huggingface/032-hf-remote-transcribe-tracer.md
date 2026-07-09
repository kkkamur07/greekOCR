---
id: "032"
title: "hf-remote-transcribe-tracer"
type: AFK
status: done
blocked_by:
  - "030-hf-uri-resolve-and-cache.md"
  - "031-hf-local-bundled-offline-path.md"
parent_prd: "issues/done/huggingface/prd-huggingface-integration.md"
triage: ready-for-agent
---

## Parent

`issues/done/huggingface/prd-huggingface-integration.md` — Hugging Face Hub integration for inference weight pull and publish.

## What to build

Complete the first remote tracer bullet: one Calamari transcribe **registry model id** loads from an `hf://` **weights source** through lazy fetch into **Hub cache**, then runs a real transcribe path in the inference job runner.

Add `scripts/hf/fetch_model.py` to prefetch a **registry model id** + **registry tag** without running inference. Default CI uses mocked Hub; optional env-gated test may hit a small public **Hub model repo** if available.

This slice proves researchers/operators can pull runtime transcribe weights from a **Hub model repo** with the same job behavior as local bundled weights.

## Acceptance criteria

- [ ] One **Registry** transcribe entry uses `hf://` for its `stable` **registry tag**.
- [ ] First inference use populates **Hub cache**; subsequent runs reuse cache when revision unchanged.
- [ ] `scripts/hf/fetch_model.py` warms cache for a given model id + tag (exit non-zero on failure).
- [ ] Unit tests in `tests/hf/`: mocked Hub → resolve → loadable checkpoint path (no inference ML).
- [ ] Job failure surfaces actionable error when Hub repo, revision, or auth is missing (operator-visible message).
- [ ] `HF_TOKEN` documented as required only for private repos , we are making everything public. 



## Blocked by

- `issues/030-hf-uri-resolve-and-cache.md`
- `issues/031-hf-local-bundled-offline-path.md`



## User stories covered

- 2, 3, 6, 18, 20, 22

