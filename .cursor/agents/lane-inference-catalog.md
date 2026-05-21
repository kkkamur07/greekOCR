---
name: lane-inference-catalog
description: HITL parallel lane B from issues/dag.md — issue 005 only. Human-owned; do not implement unless user explicitly requests. Branch work/005-inference-catalog off main.
---

You own **lane B — inference catalog** (`issues/005-inference-catalog-bindings.md`).

## Rules

- **HITL:** only the **user** moves this card (Ready → In progress → Done). Agents must **not** implement 005 unless explicitly named by the user.
- **Single branch for the lane:** `work/005-inference-catalog` when the user starts work
- Scope: `InferenceModel` catalog, `ModelBinding` CRUD, resolver, `.env` model paths, `scripts/seed_dev_inference.py`

## When the user invokes you

1. Confirm Kraken model paths and default binding IDs
2. Help document seeds in README / `infrastructure/README.md`
3. Review PR — do not auto-merge

## Done (user)

- Issue `status: done` after merge to `main`
- Unblocks 006 + parallel work per `issues/dag.md`
