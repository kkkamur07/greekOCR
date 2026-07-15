---
id: "007"
title: "annotation-vertex-edit"
type: AFK
status: review
blocked_by: []
parent_prd: "issues/prd.md"
---

## Parent

[issues/prd.md](prd.md)

## What to build

Page editor Segment editing: click vertex then Delete removes that point only; click edge adds a point; confirm before deleting a whole Segment; vertex drag must not pan the Page; Escape commits geometry and deselects (hides selection chrome / pairing strip). Add Edit undo / redo for in-session canvas steps (distinct from Annotation history).

## Acceptance criteria

- [x] Click point + Delete removes only that vertex
- [x] Click Segment edge adds a vertex
- [x] Whole-Segment delete requires confirmation
- [x] Vertex drag does not pan the Page
- [x] Escape commits + deselects (does not delete the Segment)
- [x] Edit undo / redo works for in-session canvas edits
- [x] Tests cover canvas/editor seams for these behaviors

## Blocked by

None - can start immediately
