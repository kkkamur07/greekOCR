---
name: lane-frontend-projects
description: AFK parallel lane F from issues/dag.md — issue 013 on feat/013-frontend-projects. Use when 003 APIs exist. Vertical-slice TDD via npm build + manual/API integration tests where applicable.
---

You own **lane F — frontend projects & documents** (`issues/013-frontend-projects-documents.md`).

## Rules

- **Single branch:** `feat/013-frontend-projects`
- **Base branch:** `main` or `feat/003-documents-parts-media` if document routes not on main yet
- **Blocked until:** 002, 012 done; 003 document/part APIs available
- Use generated OpenAPI types (`frontend/openapi/`); run `PYTHONPATH=. python scripts/export_openapi.py` after backend route changes on dependency branch
- JWT in local storage; Ant Design patterns from eScriptorium reuse where possible

## Deliverables (013)

Login/register, project list, document dashboard, create document, upload/reorder parts, archived hidden in default list, `npm run build` passes.

## Done

- `npm run build` green
- Issue `status: review`, `branch: feat/013-frontend-projects`
- Kanban Review ≤ 5

Do not implement 014–017 (other frontend lanes).
