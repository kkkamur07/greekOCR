---
name: lane-openapi-codegen
description: AFK parallel lane for issue 012 (OpenAPI export + openapi-typescript). Use on branch feat/012-openapi-codegen off feat/001-user-auth-jwt. Vertical-slice TDD only.
---

You own **lane E — openapi** (`issues/012-nextjs-openapi-codegen.md`).

## Rules

- Branch: `feat/012-openapi-codegen` from `feat/001-user-auth-jwt`
- **TDD/smoke**: script runs; `npm run codegen:api` succeeds; `tsc -b` passes with import of generated type
- Export via FastAPI `app.openapi()` in `scripts/export_openapi.py` (TestClient or direct `create_app()`)
- Frontend is **Vite + React** (not Next yet) — paths under `frontend/`
- Do not implement full UI pages (that's 013)

## Deliverables

1. `scripts/export_openapi.py` → `frontend/openapi/openapi.json`
2. `openapi-typescript` devDependency + `codegen:api` npm script
3. Generated types file committed (e.g. `frontend/src/api/schema.d.ts`)
4. Minimal smoke in `frontend/src/api/client.ts` or test importing `components.schemas.UserResponse`
5. Document in `frontend/README.md` or root README

## Done

- Export + codegen verified locally
- Issue `status: review`, `branch: feat/012-openapi-codegen`
- Commit; push origin

Do not touch project CRUD (002).
