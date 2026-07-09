---
id: "005"
title: "inference-catalog-bindings"
type: "HITL"
status: "done"
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

- [x] `NotFoundError` for unknown model/binding; `ValidationError` for invalid scope or task enum.

## Dev / test data

- [x] Seed at least two `InferenceModel` rows (segment + transcribe) and one project-level default binding in `scripts/seed_dev_inference.py` or extend existing seed docs.
- [x] `.env.example` entries: `DEFAULT_SEGMENT_MODEL`, `DEFAULT_TRANSCRIBE_MODEL`, optional `KRAKEN_MODEL_PATH`.
- [x] Fixture image path documented for later segment smoke tests (e.g. `backend/media/fixtures/sample_folio.png`).

## Acceptance criteria

- [x] `GET /inference/models` lists catalog entries
- [x] CRUD bindings under project/document/part; resolver returns winning model for segment and transcribe tasks
- [x] Tests: part binding overrides document overrides project
- [x] README or `infrastructure/README.md` documents how to obtain/register Kraken weights for local dev

## Blocked by

- `issues/done/000-platform-foundation.md`

## User stories addressed

- 35
- 36
- 37
- 38
