---
id: "035"
title: "hf-collection-sync"
type: AFK
status: done
blocked_by:
  - "033-hf-publish-model-from-staging.md"
parent_prd: "issues/prd-huggingface-integration.md"
triage: ready-for-agent
---

## Parent

`issues/prd-huggingface-integration.md` — Hugging Face Hub integration for inference weight pull and publish.

## What to build

Version **Hub collection** metadata in-repo and sync it to Hugging Face for discovery. Add `src/hf/collection.yaml` as source of truth for collection slug `nomos`, listing **Hub model repos** and **Hub dataset repos** by namespace/slug. Implement `scripts/hf/sync_collection.py` to push collection membership to the **Hub namespace** (mockable in CI).

## Acceptance criteria

- [ ] `src/hf/collection.yaml` lists at least one model repo and documents how to add datasets.
- [ ] Sync script updates the remote **Hub collection** from the YAML file.
- [ ] Dry-run / mock tests validate YAML schema without live Hub in default CI.
- [ ] Contributor note explains when to update the collection after publish.

## Blocked by

- `issues/033-hf-publish-model-from-staging.md`

## User stories covered

- 13, 14
