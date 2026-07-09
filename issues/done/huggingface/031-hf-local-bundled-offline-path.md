---
id: "031"
title: "hf-local-bundled-offline-path"
type: AFK
status: done
blocked_by:
  - "030-hf-uri-resolve-and-cache.md"
parent_prd: "issues/done/huggingface/prd-huggingface-integration.md"
triage: ready-for-agent
---

## Parent

`issues/done/huggingface/prd-huggingface-integration.md` — Hugging Face Hub integration for inference weight pull and publish.

## What to build

Stand up **local bundled weights** under the Hub directory layout so dev and Docker work without Hub credentials. Migrate at least one existing Calamari checkpoint (e.g. Syriac `stable`) from the interim `inference/weights/` tree into `src/hf/local/{script}/{architecture}/{model_version}/{registry_tag}/`, and point its **Registry** **weights source** at a `file://` URI relative to `src/hf/`.

Verify the full offline path: **Registry** → weight resolution → inference transcribe job loads the checkpoint and completes on bundled files alone. Update Compose volume/mount expectations so default `docker compose up` does not require `HF_TOKEN`.

## Acceptance criteria

- [ ] **Local bundled weights** layout exists under `src/hf/local/` per CONTEXT.
- [ ] At least one **Registry** entry uses `file://` relative to `src/hf/` and resolves correctly.
- [ ] Integration test runs transcribe (or equivalent architecture load) using only bundled local weights — no Hub network calls.
- [ ] Docker Compose inference services start and resolve the migrated model without Hub auth.
- [ ] Documentation in `inference/README.md` distinguishes interim `inference/weights/` from target `src/hf/local/` and states migration status.

## Blocked by

- `issues/030-hf-uri-resolve-and-cache.md`

## User stories covered

- 7, 21, 23
