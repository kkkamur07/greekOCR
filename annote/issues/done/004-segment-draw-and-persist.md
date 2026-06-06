---
id: "004"
title: "segment-draw-and-persist"
type: AFK
status: done
blocked_by:
  - "003-page-transcription-panel.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Segment drawing (polygon + rectangle), selection, autosave

## What to build

Canvas tools to draw **Polygon segment** and **Rectangle segment** (corner-based, rotatable — not axis-aligned-only) on the page image; assign **segment number** in creation order; select segment on click; autosave geometry to `data/annotations/pages/<stem>.json` via `GET/PUT` annotation API; reload segments when reopening the page.

Adapt drawing patterns from Kalamos legacy `ImageCanvas` where useful, implemented as Next.js client components.

Canvas UX defaults (rectangle gesture, vertex size, shortcuts) are agent-chosen; iterate if review feedback requires changes.

## Error handling

- [ ] Failed save shows API error; editor retains local state for retry.
- [ ] Corrupt annotation JSON returns 422 with safe empty fallback after user confirmation.

## Dev / test data

- [ ] Annotation round-trip test with polygon and rectangle fixtures.
- [ ] Manual QA checklist in issue or README for draw/select on sample folio.

## Acceptance criteria

- [ ] Draw polygon segment (4+ points); appears in overlay
- [ ] Draw rectangle segment via corner interaction; appears in overlay
- [ ] New segments receive incrementing segment numbers in creation order
- [ ] Click selects segment (visual selected state)
- [ ] Segments persist to JSON; reload page restores shapes
- [ ] Annotation store unit/integration tests pass

## Blocked by

- `issues/003-page-transcription-panel.md`

## User stories addressed

- 11
- 12
- 13
- 14
- 15
- 16
- 20
- 21
- 22
- 23
- 24
