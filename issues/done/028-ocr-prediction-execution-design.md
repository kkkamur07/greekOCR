---
id: "028"
title: "ocr-prediction-execution-design"
type: HITL
status: backlog
blocked_by:
  - "022-page-transcription-pairing-progress.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Deferred OCR prediction execution design.

## What to build

Decide how OCR prediction should execute after the manual annotation model is stable. The design should cover selected-line suggestions, Page-level prediction, Document-level prediction, job-backed versus synchronous behavior, how Model transcription layers are created, and how users accept or edit Model transcription into Ground truth.

## Acceptance criteria

- [ ] Decide whether selected-line OCR prediction is synchronous, job-backed, or both.
- [ ] Decide whether Page and Document OCR prediction use the existing job infrastructure.
- [ ] Decide how Model transcription layers are named, stored, and compared to Ground truth.
- [ ] Decide how frontend progress and errors are shown for OCR prediction.
- [ ] Decide which model runtime pieces may be packaged for the annote backend while keeping the root model workspace separate.
- [ ] Produce follow-up AFK implementation issues once the design is accepted.

## Blocked by

- `issues/022-page-transcription-pairing-progress.md`

## User stories covered

- 28
- 29
- 50
