---
id: "015"
title: "kraken-ceiling-overlay-and-source-styling"
type: AFK
status: ready
blocked_by:
  - "014-segment-refinement-auto-refine.md"
parent_prd: "issues/prd.md"
---

## Parent

`issues/prd.md` — **Kraken ceiling overlay** and **Segment source** visual distinction.

## What to build

Editor UX for refined Kraken segments — end-to-end through canvas rendering:

- **Segment source** styling: visually distinguish `manual` vs `kraken` segments on the canvas (subtle stroke/colour difference; exact styling agent-chosen).
- **Kraken ceiling overlay**: optional dashed outline of `kraken_ceiling` for the **selected** segment when `source=kraken` and ceiling is present. Toolbar toggle; **off by default**. Not shown for manual segments or when ceiling is null.
- Hand-edits to `points` do not move or hide `kraken_ceiling`; overlay reflects stored ceiling, not live clamping.
- Toggle state is session-local (not persisted to annotation JSON).

## Acceptance criteria

- [ ] Kraken segments are visually distinguishable from manual segments on the canvas
- [ ] Toolbar toggle controls **Kraken ceiling overlay** visibility; default off
- [ ] When enabled and a Kraken segment is selected, dashed outline matches stored `kraken_ceiling`
- [ ] Overlay hidden for manual segments, unselected segments, and when toggle is off
- [ ] Component or canvas tests cover toggle on/off and selected-segment behaviour

## Blocked by

- `issues/014-segment-refinement-auto-refine.md`

## User stories addressed

- 93
- 94
- 95
