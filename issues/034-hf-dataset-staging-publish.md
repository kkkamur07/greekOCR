---
id: "034"
title: "hf-dataset-staging-publish"
type: AFK
status: backlog
blocked_by:
  - "033-hf-publish-model-from-staging.md"
parent_prd: "issues/prd-huggingface-integration.md"
triage: ready-for-agent
---

## Parent

`issues/prd-huggingface-integration.md` — Hugging Face Hub integration for inference weight pull and publish.

## What to build

Add dataset publish capability separate from **Hub model repos**: stage labelled line crops (or a minimal sample corpus) under `src/hf/staging/datasets/` using the **Hub dataset slug** convention from CONTEXT (`{script}-manuscript-lines` or `{script}-{corpus}-htr-lines`). Provide a publish script that creates/updates the **Hub dataset repo** with a README describing provenance, script, and pairing conventions.

This slice does not wire automatic export from the annotation platform — it proves the staging → Hub dataset path for corpus curators.

## Acceptance criteria

- [ ] **Hub staging tree** dataset layout defined and validated.
- [ ] Publish script uploads to the correct **Hub dataset slug** under the configured **Hub namespace**.
- [ ] Dataset README states script, crop format, and relationship to future **registry model ids**.
- [ ] Dry-run / mock tests pass in default CI without live upload.
- [ ] CONTEXT distinction between **Hub model repo** and **Hub dataset repo** reflected in operator docs.

## Blocked by

- `issues/033-hf-publish-model-from-staging.md`

## User stories covered

- 12
