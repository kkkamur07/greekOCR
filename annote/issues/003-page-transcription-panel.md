---
id: "003"
title: "page-transcription-panel"
type: AFK
status: backlog
blocked_by:
  - "002-page-list-and-viewer.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Page transcription loading and text line display

## What to build

Load page transcription from `data/transcriptions/pages/<stem>.txt` via API; parse into numbered **Text lines** (split on line breaks). Show sidebar panel on the editor with the text line list. Indicate when transcription file is missing.

Includes testable **text line parser** module.

## Error handling

- [ ] Missing transcription file: API returns clear status; UI shows message (not a hard failure for opening the page).
- [ ] Empty transcription file handled gracefully.

## Dev / test data

- [ ] Sample `.txt` transcription matching sample page stem (Greek Unicode line-broken text).
- [ ] Unit tests for `split_text_lines` including Greek characters.

## Acceptance criteria

- [ ] `GET /pages/{stem}/transcription` returns raw text and parsed text lines
- [ ] Editor sidebar lists numbered text lines in order
- [ ] Page without transcription shows explicit empty/missing state
- [ ] Text line parser unit tests pass

## Blocked by

- `issues/002-page-list-and-viewer.md`

## User stories addressed

- 7
- 8
- 10
- 47
