---
id: "001"
title: "user-auth-jwt"
type: AFK
status: backlog
blocked_by:
  - "done/000-platform-foundation.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Auth API, Auth service module, Access (v1)

## What to build

Register, login, and `GET /me` with JWT and password hashing. Persist users in Postgres. Pytest covers happy path and invalid credentials with one failing tests and multiple suceeding tests. OpenAPI documents auth routes.

## Dev / test data

- [ ] Add a dev seed script or fixture (at least one test user) so integration tests and local login do not require manual register each run.

## Acceptance criteria

- [ ] `POST /auth/register` and `POST /auth/login` return JWT; passwords stored hashed
- [ ] `GET /me` returns current user when Bearer token valid; 401 when missing/invalid
- [ ] Tests: register + login + me success; wrong password fails
- [ ] Unauthenticated users cannot access protected project routes (smoke on one guarded route)

## Blocked by

- `issues/000-platform-foundation.md`

## User stories addressed

- 1
