---
id: "006"
title: "segment-text-pairing"
type: AFK
status: done
blocked_by:
  - "005-segment-edit-delete.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Manual pairing (hybrid list + inline edit)

## What to build

**Pairing** workflow: user selects a Segment first, then assigns a **Text line** by picking from the numbered list in the sidebar and/or editing text inline. Show paired vs unpaired state for both segments and text lines. Persist pairings in annotation JSON. No automatic matching.

## Error handling

- [ ] Attempting to pair a text line already used by another segment shows warning and blocks or requires explicit override (pick one policy; default: block double-pairing).
- [ ] Unpairing clears pairing without deleting segment geometry.

## Dev / test data

- [ ] Sample page with 3+ text lines and 2+ segments for manual QA.
- [ ] Annotation tests for pairing round-trip.

## Acceptance criteria

- [ ] Select segment → pick text line from list → pairing saved
- [ ] Inline edit updates paired text (stored as pairing override or text line index per design)
- [ ] UI shows paired/unpaired segments and text lines
- [ ] Pairings reload on page reopen
- [ ] No automatic alignment between segments and text

## Blocked by

- `issues/005-segment-edit-delete.md`

## User stories addressed

- 25
- 26
- 27
- 28
- 29
- 30
- 31
