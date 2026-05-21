---
id: "011"
title: "access-public-published"
type: AFK
status: backlog
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

- [ ] Access policy raises `AccessDeniedError`; handlers from 001 return 403 with consistent body.

## Dev / test data

- [ ] Seed: one `published` and one `draft` document under dev project; anonymous pytest client fixture (no Bearer).
- [ ] Public slug or id documented for manual browser check.

## Acceptance criteria

- [ ] Unauthenticated `GET` published document + parts + layout + layers succeeds
- [ ] Unauthenticated `POST` segment/transcribe/layout PATCH on published returns 403
- [ ] Project member can still edit and run jobs on published document
- [ ] Draft document not public readable
- [ ] Tests per PRD access policy table

## Blocked by

- `issues/003-documents-parts-media.md`
- `issues/done/001-user-auth-jwt.md`

## User stories addressed

- 6
- 7
- 8
