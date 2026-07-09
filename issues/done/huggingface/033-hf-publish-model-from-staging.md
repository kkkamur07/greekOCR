---
id: "033"
title: "hf-publish-model-from-staging"
type: AFK
status: done
blocked_by:
  - "032-hf-remote-transcribe-tracer.md"
parent_prd: "issues/done/huggingface/prd-huggingface-integration.md"
triage: ready-for-agent
---

## Parent

`issues/done/huggingface/prd-huggingface-integration.md` — Hugging Face Hub integration for inference weight pull and publish.

## What to build

Enable repeatable publish of inference checkpoints from the **Hub staging tree** to a **Hub model repo**. Training or manual copy places **Hub artifacts** under `src/hf/staging/models/{script}/{architecture}/{model_version}/{registry_tag}/` (Calamari SavedModel + JSON sidecar per CONTEXT). A publish script uploads to `{namespace}/{hub-repo-slug}`, tags the **Hub revision** to match the **registry tag**, and writes/updates a model card describing script, architecture, and **registry model id**.

Dry-run and validation modes must work without live upload in default CI (mock Hub API). Document the operator runbook for namespace, token, and first publish.

## Acceptance criteria

- [ ] **Hub staging tree** model layout documented and validated before upload.
- [ ] Publish script creates or updates the target **Hub model repo** with correct **Hub repo slug** convention.
- [ ] Published revision is addressable as `hf://<namespace>/<hub-repo-slug>@<registry-tag>`.
- [ ] Model card includes script, architecture, task, and **registry model id**.
- [ ] Dry-run / mock tests cover staging validation without requiring upload credentials in CI.
- [ ] Runbook section covers first-time publish steps (token, namespace, staging → push).

## Blocked by

- `issues/032-hf-remote-transcribe-tracer.md`

## User stories covered

- 9, 10, 11, 24
