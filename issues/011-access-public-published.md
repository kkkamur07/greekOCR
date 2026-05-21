---
id: "011"
title: "access-public-published"
type: AFK
status: review
branch: feat/011-access-public
blocked_by:
  - "003-documents-parts-media.md"
  - "done/001-user-auth-jwt.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Access policy module, Public view, Published behavior

## What to build

**Access policy** enforcing: project members read/write; anonymous or non-member read **published** documents only; no mutate or job enqueue for non-members on published; members retain full rights on published. Public GET routes for document, parts, layout, transcriptions (read-only).

## Error handling

- [x] Access policy raises `AccessDeniedError`; handlers from 001 return 403 with consistent body.

## Dev / test data

- [x] `tests/test_access_public.py` builds published + draft via live API; anonymous client (no Bearer).
- [ ] Public slug or id documented for manual browser check (use project UUID from seed).

## Acceptance criteria

- [x] Unauthenticated `GET` published document + parts + layout + layers succeeds
- [x] Unauthenticated `POST`/`PATCH` on member routes returns 401
- [x] Project member can still edit published document
- [x] Draft document not public readable (404)
- [x] Tests per PRD access policy table (`tests/test_access_public.py`, `tests/test_document_access.py`)

## Blocked by

- `issues/003-documents-parts-media.md`
- `issues/done/001-user-auth-jwt.md`

## User stories addressed

- 6
- 7
- 8
