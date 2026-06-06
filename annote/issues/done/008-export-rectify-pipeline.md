---
id: "008"
title: "export-rectify-pipeline"
type: AFK
status: done
blocked_by:
  - "007-export-dirty-state.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Processing pipeline (rectify), export paired outputs, warnings

## What to build

Implement pluggable **Processing** pipeline with v1 step `rectify` only (polygon/rectangle → axis-aligned line image). **Export service** writes paired segments to `data/manuscripts/lines/<stem>_<segment_number>.jpg` and `data/transcriptions/lines/<stem>_<segment_number>.txt`; skips unpaired segments; returns warnings for unpaired segments and unused text lines. Wire `POST /pages/{stem}/export` to full pipeline; clear dirty state on success.

Processing and export service must be testable without the UI.

## Error handling

- [ ] Export with zero paired segments returns warnings, writes no line files (or documents behavior).
- [ ] Partial export failure reports which segment numbers failed; does not mark fully exported unless configured policy says otherwise.
- [ ] Re-export overwrites previous line files for same stem/segment number.

## Dev / test data

- [ ] Synthetic page image + rectangle segment fixture for rectify unit test.
- [ ] Integration test: two paired segments → two jpg + two txt on disk.
- [ ] Greek text in exported `.txt` round-trips UTF-8.

## Acceptance criteria

- [ ] `rectify` produces reasonable axis-aligned crop for polygon and rectangle segments
- [ ] Export writes correctly named `.jpg` and `.txt` for each paired segment only
- [ ] Response includes warnings for unpaired segments and unused text lines
- [ ] Export clears dirty state; subsequent edit sets dirty again
- [ ] Processing pipeline accepts step list; only `rectify` implemented
- [ ] Rectify and export service tests pass without frontend

## Blocked by

- `issues/007-export-dirty-state.md`

## User stories addressed

- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41
- 42
- 43
- 45
