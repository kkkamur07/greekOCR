---
id: "020"
title: "document-line-transcription-model"
type: "AFK"
status: "done"
blocked_by:
  - "018-annote-production-root.md"
parent_prd: "issues/prd-annote-merge.md"
---



## Parent

`issues/prd-annote-merge.md` — Canonical Document, Document part, Line, Transcription, and Line transcription model.

## What to build

Adapt the document context so a Document owns document-level Transcriptions, each Document part represents one Page, annote Segments persist as Lines, and approved per-line text persists as Line transcriptions in the Ground truth layer. Add a boolean Review status on each Document part without introducing a separate Segment table.

## Acceptance criteria

- [ ] A Document has one canonical Ground truth Transcription layer that can span multiple Document parts.
- [ ] A Document part can persist Lines with Segment geometry, order, source metadata, and optional Kraken ceiling data.
- [ ] A Line transcription links one Line to one document-level Transcription.
- [ ] Approved text is stored per Line, not as a canonical Page text blob.
- [ ] Review status is stored as a boolean on the Document part and defaults to unreviewed.
- [ ] Model transcription remains separate from Ground truth transcription in the model.
- [ ] Database migrations and API schema tests cover the new or adapted fields and relationships.

## Blocked by

- `issues/018-annote-production-root.md`

## User stories covered

- 6
- 7
- 16
- 25
- 26
- 27
- 28
- 32
- 33
