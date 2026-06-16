---
id: "025"
title: "export-approved-line-artifacts"
type: AFK
status: review
blocked_by:
  - "022-page-transcription-pairing-progress.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Export Processed line images and Line transcription files from approved text.

## What to build

Port Export into the annotation workflow so a project member can export current approved Line transcriptions and Segment geometry for a Document part/Page as training-ready Processed line images and Line transcription files. Export should warn about unpaired Segments and unused Text lines without becoming its own durable business object.

## Acceptance criteria

- [x] A project member can trigger Export for a Document part/Page.
- [x] Export uses current Line geometry and approved Ground truth Line transcription text.
- [x] Each paired Line can produce one Processed line image and one Line transcription file.
- [x] Export skips unpaired Lines and reports a warning.
- [x] Export reports unused candidate Text lines when relevant.
- [x] Export does not require migrating or modifying existing local data contents.
- [x] API/service tests cover artifact naming, paired output, warnings, and access control.

## Blocked by

- `issues/022-page-transcription-pairing-progress.md`

## User stories covered

- 43
- 44
- 45
