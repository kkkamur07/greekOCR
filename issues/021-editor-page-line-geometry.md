---
id: "021"
title: "editor-page-line-geometry"
type: AFK
status: review
blocked_by:
  - "019-authenticated-platform-shell.md"
  - "020-document-line-transcription-model.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Annote editor for Document part/Page Line geometry.

## What to build

Port annote's editor theme and Page canvas workflow into the merged frontend so a project member can open a Document part/Page, view the Page image, draw Polygon and Rectangle Segments, edit Segment geometry, delete Segments, and persist those edits as Lines in the document context.

## Acceptance criteria

- [ ] A project member can open a Document part/Page from the authenticated Document flow.
- [ ] The editor displays the Page image with annote's visual theme.
- [ ] A user can create Polygon and Rectangle Segments on the Page.
- [ ] Segment geometry saves as Line geometry through authenticated API calls.
- [ ] A user can edit and delete existing Lines/Segments.
- [ ] Unauthorized users cannot read or modify Page geometry.
- [ ] UI and API tests cover create, update, delete, and reload of Line geometry.

## Blocked by

- `issues/019-authenticated-platform-shell.md`
- `issues/020-document-line-transcription-model.md`

## User stories covered

- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 41
- 42
