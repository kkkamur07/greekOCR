---
id: "005"
title: "segment-edit-delete"
type: AFK
status: backlog
blocked_by:
  - "004-segment-draw-and-persist.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Segment vertex editing and deletion

## What to build

Edit mode for selected segment: drag vertices for polygon and rectangle; delete selected segment (with confirmation or undo-friendly pattern). Changes autosave to annotation JSON. Segment numbers already assigned are not reused after delete (stable numbering policy per PRD).

## Error handling

- [ ] Deleting selected segment clears selection in UI.
- [ ] Save failure surfaces toast/banner; geometry not silently dropped.

## Dev / test data

- [ ] Integration test: create segments via API, update points, delete one, verify JSON on disk.

## Acceptance criteria

- [ ] Move polygon vertices; persisted after save
- [ ] Move rectangle corners; persisted after save
- [ ] Delete segment removes from canvas and JSON
- [ ] Reopen page reflects edits
- [ ] API/annotation tests cover update and delete

## Blocked by

- `issues/004-segment-draw-and-persist.md`

## User stories addressed

- 17
- 18
- 19
