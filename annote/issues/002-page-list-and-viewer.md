---
id: "002"
title: "page-list-and-viewer"
type: AFK
status: backlog
blocked_by:
  - "001-foundation-scaffold.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Page list, page image display, navigation

## What to build

Vertical slice: discover page images from `data/manuscripts/pages/`, expose `GET /pages` and `GET /pages/{stem}/image`, build Next.js home **Page list** (sorted filenames) and `/pages/[stem]` editor route with full-page image display, zoom, and pan. Back link to list.

No segments or transcriptions yet.

## Error handling

- [ ] Empty pages directory shows empty state, not a crash.
- [ ] Unknown stem returns 404.

## Dev / test data

- [ ] At least one JPEG in `data/manuscripts/pages/` (sample Harmenopulus image).
- [ ] Backend test: `list_pages` finds fixture file.

## Acceptance criteria

- [ ] Home lists all page images sorted by filename
- [ ] Clicking a page opens editor with correct image rendered
- [ ] Zoom and pan work on large folio image
- [ ] Navigate back to page list
- [ ] API integration test with temp data directory

## Blocked by

- `issues/001-foundation-scaffold.md`

## User stories addressed

- 2
- 3
- 4
- 5
- 6
- 9
