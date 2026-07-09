---
id: "017"
title: "frontend-public-published-view"
type: AFK
status: done
blocked_by:
  - "done/011-access-public-published.md"
  - "done/012-nextjs-openapi-codegen.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Public view for published documents

## What to build

Unauthenticated **Public view** route: read published document metadata, parts, layout overlay, and transcription layers (read-only). No edit controls or job enqueue. Published workflow badge; draft/archived not reachable without login.

## Error handling

- [ ] 404 for draft slug; generic message for forbidden mutations (should not appear in UI).

## Dev / test data

- [ ] Seed one `published` document with public-readable slug; document URL pattern in README (e.g. `/public/documents/{id}`).
- [ ] Integration test credentials: anonymous client fixture in `tests/conftest.py` (no auth headers).

## Acceptance criteria

- [ ] Anonymous user opens published document and sees layout + layers
- [ ] No save/segment/transcribe buttons in public view
- [ ] Project member opening same URL still has link to full editor
- [ ] Draft document public URL returns 404

## Blocked by

- `issues/done/011-access-public-published.md`
- `issues/done/012-nextjs-openapi-codegen.md`

## User stories addressed

- 6
- 7
- 9
- 10
