---
id: "015"
title: "frontend-transcription-editor"
type: AFK
status: done
blocked_by:
  - "010-ground-truth-copy-edit-api.md"
  - "done/012-nextjs-openapi-codegen.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Transcription edit mode, layer picker, copy to ground truth

## What to build

**Transcription edit** mode: layer picker (model layers read-only for typing); edit text only on **Ground truth**; **Copy to ground truth** action (selection or whole page). Optional side-by-side compare view (read-only on model layers).

## Error handling

- [x] Prevent editing model layer in UI; API errors on illegal PATCH surfaced to user.

## Dev / test data

- [x] Requires seeded document with ground truth + model layer (010 seed); link from `scripts/README` or frontend README.

## Acceptance criteria

- [x] User switches to transcription mode and edits ground truth line text
- [x] Copy from model layer populates ground truth for selected lines
- [x] Model layer text not editable in transcription edit mode
- [x] Build passes; smoke test documented in frontend README

## Blocked by

- `issues/010-ground-truth-copy-edit-api.md`
- `issues/done/012-nextjs-openapi-codegen.md`

## User stories addressed

- 32
- 33
