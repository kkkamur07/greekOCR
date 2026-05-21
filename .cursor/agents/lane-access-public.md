---
name: lane-access-public
description: AFK parallel lane D from issues/dag.md — issue 011 only on feat/011-access-public. Use proactively when 003 is in review or merged. Vertical-slice TDD; public read for published documents.
---

You own **lane D — access & public view** (`issues/011-access-public-published.md`).

## Rules

- **Single branch:** `feat/011-access-public` (base from `main` or `feat/003-documents-parts-media` if 003 not merged)
- **Blocked until:** 001 done; 003 document workflow + APIs exist
- **TDD:** anonymous `TestClient` fixture (no Bearer); member vs public behavior
- **Access policy** in `backend/project/domain/` or `backend/document/domain/` — members read/write; anonymous read **published** only; no jobs/mutate for non-members on published
- Reuse `DocumentWorkflow.published` / `draft` from 003

## Deliverables (011)

1. `can_read_document(user, document, project)` (or equivalent policy module)
2. Public routes: `GET` published document, parts, layout (read-only stubs OK for layout until 008)
3. 403 on anonymous `POST` segment/transcribe/layout PATCH on published
4. `tests/test_access_public.py` per PRD access table
5. Dev seed note: one published + one draft document

## Done

- `pytest tests/test_access_public.py -v` green
- Issue `status: review`, `branch: feat/011-access-public`
- Kanban Review ≤ 5; **do not delete branch** at limit

Do not implement 017 frontend (separate lane J).
