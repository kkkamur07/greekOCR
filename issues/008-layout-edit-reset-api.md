---
id: "008"
title: "layout-edit-reset-api"
type: AFK
status: review
blocked_by:
  - "007-segment-merge.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Layout API (blocks/lines CRUD, reset layout)

## What to build

Authenticated layout routes: CRUD **Block** and **Line** under a part (baseline/mask/box JSON); saves set `manual_geometry=true`. **Reset layout** clears manual flag on selected lines or whole part so next segment can replace geometry. List layout for editor canvas.

## Error handling

- [ ] `AccessDeniedError` for non-members; `NotFoundError` for unknown part/line ids.

## Dev / test data

- [ ] Extend document seed: one part with at least one block and two lines (one manual, one machine) for layout API tests.
- [ ] Export sample geometry JSON in `tests/fixtures/layout_line.json` matching eScriptorium-compatible shape.

## Acceptance criteria

- [ ] Member can PATCH line baseline and block box; `manual_geometry` set true
- [ ] Reset layout endpoint clears manual flag on targeted lines
- [ ] Non-member cannot mutate layout
- [ ] Tests: save manual line survives re-segment (with 007 wired)

## Blocked by

- `issues/007-segment-merge.md`

## User stories addressed

- 18
- 19
- 23
