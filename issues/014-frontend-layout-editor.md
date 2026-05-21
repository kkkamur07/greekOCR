---
id: "014"
title: "frontend-layout-editor"
type: HITL
status: backlog
blocked_by:
  - "008-layout-edit-reset-api.md"
  - "done/012-nextjs-openapi-codegen.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Layout edit mode, canvas (Fabric), geometry JSON

## What to build

Document editor **Layout edit** mode: canvas (Fabric or equivalent) rendering part image, blocks, and lines; draw/edit baselines and boxes; save via layout API; toolbar toggle vs transcription mode. Port eScriptorium geometry interaction concepts (not Vue code).

**Human decisions required:** canvas UX (tools, shortcuts), performance target for large folios, and default zoom/pan behavior.

## Error handling

- [ ] Failed save shows API error; optimistic rollback on PATCH failure.

## Dev / test data

- [ ] Use seeded part image from 003/008; document sample dimensions in `frontend/README.md`.
- [ ] Optional static demo part in `frontend/public/demo-folio.jpg` for offline UI iteration.

## Acceptance criteria

- [ ] Toggle layout mode; edit line baseline; save sets manual geometry (verified via API refetch)
- [ ] Reset layout control calls API and updates canvas
- [ ] Member-only access enforced by API (UI handles 403)

## Blocked by

- `issues/008-layout-edit-reset-api.md`
- `issues/done/012-nextjs-openapi-codegen.md`

## User stories addressed

- 18
- 19
- 23
- 49
