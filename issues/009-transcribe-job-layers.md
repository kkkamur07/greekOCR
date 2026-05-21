---
id: "009"
title: "transcribe-job-layers"
type: AFK
status: backlog
blocked_by:
  - "007-segment-merge.md"
  - "005-inference-catalog-bindings.md"
  - "done/003-documents-parts-media.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Transcribe job, transcription layer factory, canonical transcribe DTO

## What to build

`POST` transcribe on part/document → **Job**; adapter returns **CanonicalTranscribeResult**; **Transcription layer factory** creates a new model layer per job (never writes ground truth). Persist **LineTranscription** with confidence. `GET` lists transcription layers for a document.

## Error handling

- [ ] Missing lines/layout → `ValidationError` or 409 with clear message; job errors sanitized on `Job.error`.

## Dev / test data

- [ ] Seed document with lines (from layout seed) and empty **Ground truth** layer created by factory on first open.
- [ ] Mock transcribe adapter for CI returning fixed line texts in `tests/fixtures/transcribe_canonical.json`.

## Acceptance criteria

- [ ] Each transcribe job creates a distinct model **Transcription** layer
- [ ] Ground truth layer exists once per document and stays empty until 010
- [ ] Job enqueue + poll returns line texts on model layer
- [ ] Tests: two transcribe jobs → two model layers; ground truth unchanged

## Blocked by

- `issues/007-segment-merge.md`
- `issues/005-inference-catalog-bindings.md`
- `issues/done/003-documents-parts-media.md`

## User stories addressed

- 26
- 27
- 28
- 29
- 30
- 34
- 40
