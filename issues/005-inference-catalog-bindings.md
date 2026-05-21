---
id: "005"
title: "inference-catalog-bindings"
type: HITL
status: ready
blocked_by:
  - "done/000-platform-foundation.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — InferenceModel catalog, ModelBinding resolver, ModelAdapter registry

## What to build

Register **InferenceModel** rows (Kraken segment, Kraken/HTR transcribe, future TrOCR/HF) with artifact paths and default params. **ModelBinding** CRUD at project, document, and document-part scope with “most specific wins” resolution. Stub **ModelAdapter** interface and registry keys (implementations can noop until 006/009). Document where model weights live on disk/GPU for local dev.

**Human decisions required:** default model IDs in `.env`, Kraken model file locations, and which bindings ship in dev seed.

## Error handling

- [ ] `NotFoundError` for unknown model/binding; `ValidationError` for invalid scope or task enum.

## Dev / test data

- [ ] Seed at least two `InferenceModel` rows (segment + transcribe) and one project-level default binding in `scripts/seed_dev_inference.py` or extend existing seed docs.
- [ ] `.env.example` entries: `DEFAULT_SEGMENT_MODEL`, `DEFAULT_TRANSCRIBE_MODEL`, optional `KRAKEN_MODEL_PATH`.
- [ ] Fixture image path documented for later segment smoke tests (e.g. `backend/media/fixtures/sample_folio.png`).

## Acceptance criteria

- [ ] `GET /inference/models` lists catalog entries
- [ ] CRUD bindings under project/document/part; resolver returns winning model for segment and transcribe tasks
- [ ] Tests: part binding overrides document overrides project
- [ ] README or `infrastructure/README.md` documents how to obtain/register Kraken weights for local dev

## Blocked by

- `issues/done/000-platform-foundation.md`

## User stories addressed

- 35
- 36
- 37
- 38
