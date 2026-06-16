---
id: "024"
title: "annotation-history-restore"
type: AFK
status: review
blocked_by:
  - "022-page-transcription-pairing-progress.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Compact Annotation history in the annotation bounded context.

## What to build

Implement Annotation history as compact restorable Page annotation states in the annotation context. A researcher can list History snapshots for a Document part/Page and restore one to replace current Line geometry and Ground truth Line transcriptions.

## Acceptance criteria

- [x] History snapshots are stored in the annotation context and reference a Document part/Page.
- [x] Snapshots contain compact restorable annotation state, excluding images, generated exports, and raw edit-by-edit events.
- [x] Snapshots capture enough Line geometry and Ground truth Line transcription state to restore a Page.
- [x] A project member can list snapshots for a Page.
- [x] A project member can restore a snapshot and see the editor update after reload.
- [x] Snapshot retention is bounded so storage growth is controlled.
- [x] API/service tests cover snapshot creation, listing, restore, access control, and retention.

## Blocked by

- `issues/022-page-transcription-pairing-progress.md`

## User stories covered

- 37
- 38
- 39
- 40
