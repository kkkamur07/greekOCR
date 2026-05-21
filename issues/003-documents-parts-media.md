---
id: "003"
title: "documents-parts-media"
type: AFK
status: review
branch: feat/003-documents-parts-media
blocked_by:
  - "done/002-projects-sharing.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Documents/Parts API, Media store, Document workflow

## What to build

Documents under projects; ordered DocumentParts with image upload and media store; workflow field `draft` | `published` | `archived`; part reorder and delete. Document dashboard API lists parts. Archived documents excluded from default list.

## Error handling

- [x] Use `NotFoundError`, `AccessDeniedError`, `ValidationError` from `backend.core.exceptions` in document/media use cases.

## Dev / test data

- [x] `tests/test_documents.py` uses `MINIMAL_PNG` bytes + `owner_project` fixture (live register + project create).

## Acceptance criteria

- [x] CRUD documents under project; only project members can access
- [x] Upload image creates DocumentPart with stable order; reorder and delete work
- [x] Media served or URL returned for part image
- [x] Workflow transitions persist; archived hidden from default document list
- [x] Tests: member CRUD parts; non-member denied

## Blocked by

- `issues/done/002-projects-sharing.md`

## User stories addressed

- 9
- 10
- 11
- 12
- 14
- 15
