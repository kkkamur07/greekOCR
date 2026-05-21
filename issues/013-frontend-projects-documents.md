---
id: "013"
title: "frontend-projects-documents"
type: AFK
status: review
branch: feat/013-frontend-projects
blocked_by:
  - "003-documents-parts-media.md"
  - "done/012-nextjs-openapi-codegen.md"
  - "done/002-projects-sharing.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Frontend routes: project list, document dashboard

## What to build

Vite/React routes: login/register, project list (owned + shared), project dashboard, document list with workflow badges, create document, upload/reorder parts (uses generated OpenAPI types). Wire `src/api/client.ts` with JWT from local storage.

## Error handling

- [ ] Surface API `detail` messages in Ant Design notifications; 401 → redirect to login.

## Dev / test data

- [ ] Document in README: run `scripts/seed_dev_user.py`, create project via API or optional `seed_dev_workspace.py` with one project + document.
- [ ] `.env.local.example` for `VITE_API_BASE_URL=http://localhost:8000`.

## Acceptance criteria

- [ ] User can log in and see only their projects
- [ ] Create project and document; upload at least one part image
- [ ] Archived documents hidden from default list in UI
- [ ] `npm run build` passes with generated types

## Blocked by

- `issues/003-documents-parts-media.md`
- `issues/done/012-nextjs-openapi-codegen.md`
- `issues/done/002-projects-sharing.md`

## User stories addressed

- 1
- 2
- 3
- 4
- 11
- 12
- 14
- 15
