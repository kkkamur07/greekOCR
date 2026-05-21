---
id: "010"
title: "ground-truth-copy-edit-api"
type: AFK
status: backlog
blocked_by:
  - "009-transcribe-job-layers.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Copy to ground truth, transcription edit API

## What to build

**Copy to ground truth** use case (whole document or selected line ids). PATCH ground truth line texts only. Model layers read-only via API except copy source. List layers with kind `ground_truth` | `model`.

## Error handling

- [ ] Copy from unknown layer → `NotFoundError`; patch model layer directly → `AccessDeniedError` or 409.

## Dev / test data

- [ ] Seed: document with ground truth + one model layer containing sample line transcriptions for copy tests.
- [ ] Fixture line id list for partial copy integration test.

## Acceptance criteria

- [ ] Copy overwrites ground truth text for selected lines
- [ ] PATCH ground truth line text persists; PATCH model layer rejected
- [ ] Tests: copy whole doc; copy subset; direct ground truth edit

## Blocked by

- `issues/009-transcribe-job-layers.md`

## User stories addressed

- 31
- 32
