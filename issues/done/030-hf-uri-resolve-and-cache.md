---
id: "030"
title: "hf-uri-resolve-and-cache"
type: AFK
status: done
blocked_by: []
parent_prd: "issues/prd-huggingface-integration.md"
triage: ready-for-agent
---

## Parent

`issues/prd-huggingface-integration.md` — Hugging Face Hub integration for inference weight pull and publish.

## What to build

Introduce **Hub integration** as the first tracer bullet for remote weights: parse `hf://<namespace>/<hub-repo-slug>@<registry-tag>` **weights sources**, download **Hub artifacts** into the **Hub cache** at `src/hf/cache/<registry_model_id>/<registry_tag>/`, and record a **Hub cache manifest** so reuse requires a matching **Hub revision** hash (not just an existing directory).

Wire this into inference weight resolution so existing `file://` and `package://` behavior is unchanged, but a **Registry** entry may optionally point at `hf://`. Provide a thin Hub client wrapper (via `huggingface_hub`) mockable in tests.

End-to-end for this slice: given a mocked ( think you should upload it and test it live ) Hub response, resolving an `hf://` URI yields a local directory path with the expected **Hub artifact** files; a second resolve with the same revision skips re-download; a changed revision refreshes the cache.

## Acceptance criteria

- [ ] `hf://` URIs parse per `inference/CONTEXT.md` (`namespace`, **hub-repo-slug**, **registry tag**).
- [ ] Invalid or unsupported schemes still fail clearly; `file://` and `package://` tests remain green.
- [ ] **Hub cache** layout and **Hub cache manifest** match CONTEXT conventions.
- [ ] Cache hit avoids re-download when manifest matches remote revision; cache miss refreshes artifacts.
- [ ] Unit tests mock Hub at the download/metadata boundary (no live Hub required in default CI).
- [ ] Inference weight resolver delegates `hf://` to **Hub integration** without changing job HTTP contracts.



## Blocked by

None — can start immediately.

## User stories covered

- 1, 4, 5, 8, 19

