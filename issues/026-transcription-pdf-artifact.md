---
id: "026"
title: "transcription-pdf-artifact"
type: AFK
status: backlog
blocked_by:
  - "022-page-transcription-pairing-progress.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Transcription PDF review artifact.

## What to build

Port the Transcription PDF artifact into the merged annotation workflow. A researcher can generate a single-page PDF for a Document part/Page that places paired Ground truth text at Segment positions for review or sharing, without introducing Page lock behavior.

## Acceptance criteria

- [ ] A project member can generate a Transcription PDF for a Document part/Page.
- [ ] The PDF uses current paired Ground truth Line transcriptions and Segment geometry.
- [ ] Unpaired Lines are omitted from the PDF.
- [ ] A Page with no paired Lines can still generate a blank same-size PDF.
- [ ] PDF generation does not depend on Page lock.
- [ ] API/service tests cover generated content behavior, empty output, and access control.

## Blocked by

- `issues/022-page-transcription-pairing-progress.md`

## User stories covered

- 46
