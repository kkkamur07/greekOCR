---
id: "012"
title: "nextjs-openapi-codegen"
type: AFK
status: done
branch: feat/012-openapi-codegen
blocked_by:
  - "done/001-user-auth-jwt.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — OpenAPI export drives frontend type codegen

## What to build

Export FastAPI OpenAPI schema to `frontend/openapi/openapi.json` and generate TypeScript types for the Vite/React client (`openapi-typescript`). Document the workflow in README or `frontend/README.md`.

Pipeline: `scripts/export_openapi.py` → `frontend/openapi/openapi.json` → `npm run codegen:api` → `frontend/src/api/schema.d.ts`.

## Dev / test data

- [x] Committed `frontend/openapi/openapi.json` and `schema.d.ts` for CI; regen documented in `frontend/README.md`.
- [x] `sort_keys=True` on export for stable git diffs.

## Acceptance criteria

- [x] `scripts/export_openapi.py` (or Makefile target) writes `frontend/openapi/openapi.json` from `create_app()` schema
- [x] `frontend/package.json` has `codegen:api` script using `openapi-typescript`
- [x] Generated types include auth routes (`TokenResponse`, `UserResponse`) after 001 is on the branch
- [x] Smoke: TypeScript build passes with a minimal import of a generated type
- [x] `.gitignore` does not ignore generated types if they are committed (or document regen-only — prefer committed for CI)

## Blocked by

- `issues/done/001-user-auth-jwt.md`

## User stories addressed

- 48
