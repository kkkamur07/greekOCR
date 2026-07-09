---
id: "018"
title: "annote-production-root"
type: "AFK"
status: "done"
blocked_by: "[]"
parent_prd: "issues/prd-annote-merge.md"
---



## Parent

`issues/prd-annote-merge.md` — Annote production app root and platform relocation.

## What to build

Relocate the production platform spine into annote so backend, frontend, and infrastructure run from the annote app root while the root model workspace remains separate. Preserve the current database-backed auth/project/document behavior after relocation, and keep existing local data contents untouched.

## Acceptance criteria

- [x] The merged backend starts from the annote app root and exposes the existing health, auth, project, document, media, and OpenAPI surfaces.
- [x] The merged frontend starts from the annote app root and can be built against the merged OpenAPI schema.
- [x] Infrastructure assets needed for database migrations and local app startup live under the annote app root.
- [x] The root model workspace remains at repository root and is not moved.
- [x] Existing data contents are not modified or migrated.
- [x] Backend and frontend smoke tests/build checks pass from their new annote locations.

## Blocked by

None - can start immediately.

## User stories covered

- 1
- 2
- 5
- 9
- 10
- 12
- 47
- 49
