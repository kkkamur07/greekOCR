---
id: "011"
title: "transcription-pdf-share"
type: AFK
status: done
blocked_by:
  - "009-page-lock.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Transcription PDF share mode: frozen PDF at lock, separate Preview vs Share buttons

## What to build

Complete the two-mode **Transcription PDF** UX. Live **preview** PDF already exists (`GET .../transcription.pdf`); add **share** PDF written to disk when the page is **locked**, served via a dedicated download route, and split the UI into two buttons.

- On lock: generate and write `<stem>_transcription.pdf` under `data/manuscripts/share/` from annotation at lock time.
- On unlock: invalidate or remove share PDF so stale share files are not served.
- `GET /pages/{stem}/transcription.share.pdf` (or equivalent) returns frozen file only when page is locked and file exists; 404 otherwise.
- Editor: **Preview PDF** always enabled (live); **Share PDF** enabled only when locked.
- Nested **`TranscriptionPdfSettings`** for share directory and filename pattern.
- Paired segments only in both modes (no unpaired markers in v1).

Does not duplicate live preview implementation; extends it with share persistence and UI split.

## Acceptance criteria

- [x] Locking a page writes a share PDF to the configured share directory.
- [x] Share PDF bytes match annotation at lock time; changing annotation after unlock does not alter the on-disk share file from the previous lock until re-locked.
- [x] Share download endpoint returns 404 when page is unlocked or share file missing.
- [x] Preview PDF endpoint still regenerates from current annotation on every request.
- [x] Editor shows separate Preview PDF and Share PDF actions with correct enabled/disabled states.
- [x] Greek Unicode text appears in both preview and share PDFs (existing font behaviour preserved).
- [x] `TranscriptionPdfSettings` nested in `Settings`.
- [x] Backend tests: share written at lock; preview vs share content diverges after unlock + edit + re-preview.

## Blocked by

- `issues/009-page-lock.md` (share PDF is written at lock; Share button gated on locked state)

## User stories addressed

- User story 48
- User story 49
- User story 50
- User story 51
- User story 52
- User story 53
- User story 54
- User story 75 (TranscriptionPdfSettings portion)
