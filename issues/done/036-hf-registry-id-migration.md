---
id: "036"
title: "hf-registry-id-migration"
type: AFK
status: done
blocked_by:
  - "032-hf-remote-transcribe-tracer.md"
parent_prd: "issues/prd-huggingface-integration.md"
triage: ready-for-agent
---

## Parent

`issues/prd-huggingface-integration.md` — Hugging Face Hub integration for inference weight pull and publish.

## What to build

Migrate legacy **registry model ids** (`greek-calamariv1`, `syriac-calamariv1`, `kraken-blla`) to the `{script}-{architecture}-{model_version}` convention from CONTEXT (e.g. `greek-calamari-v1`, `syriac-calamari-v1`, and an equivalent segment id). Update `registry.yaml`, **Hub cache** paths, dev platform **InferenceModel** seeds, and inference integration tests so platform `registry://` **artifact_ref** values and inference **Registry** stay aligned.

Prefer a single cutover with a short compatibility note over indefinite dual ids, unless a temporary alias is required for one release.

## Acceptance criteria

- [x] `registry.yaml` uses new **registry model ids**; **weights sources** point at `src/hf/local/` and/or `hf://` as appropriate.
- [x] Platform dev seed (`seed_dev_inference.py` or equivalent) references the new ids.
- [x] Inference and platform integration tests updated; no dangling references to legacy ids in active code paths.
- [x] Migration note documents old → new id mapping for operators.
- [x] **Hub repo slug** and **registry model id** relationship matches CONTEXT.

## Blocked by

- `issues/032-hf-remote-transcribe-tracer.md`

## User stories covered

- 15, 16, 17
