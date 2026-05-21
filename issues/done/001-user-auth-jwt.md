---
id: "001"
title: "user-auth-jwt"
type: AFK
status: done
branch: feat/001-user-auth-jwt
blocked_by:
  - "done/000-platform-foundation.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Auth API, Auth service module, Access (v1)

## What to build

Register, login, and `GET /me` with JWT and password hashing. Persist users in Postgres. Pytest covers happy path and invalid credentials with one failing tests and multiple suceeding tests. OpenAPI documents auth routes.

## Error handling

- [x] Register FastAPI exception handlers mapping `backend.core.exceptions` (`NotFoundError` → 404, `AccessDeniedError` → 403, `ValidationError` → 422, etc.).

## Dev / test data

- [x] Add a dev seed script or fixture (at least one test user) so integration tests and local login do not require manual register each run.

## Acceptance criteria

- [x] `POST /auth/register` and `POST /auth/login` return JWT; passwords stored hashed
- [x] `GET /me` returns current user when Bearer token valid; 401 when missing/invalid
- [x] Tests: register + login + me success; wrong password fails
- [x] Unauthenticated users cannot access protected project routes (smoke on one guarded route)

## Blocked by

- `issues/done/000-platform-foundation.md`

## User stories addressed

- 1
