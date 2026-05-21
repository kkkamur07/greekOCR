---
id: "002"
title: "projects-sharing"
type: AFK
status: backlog
blocked_by:
  - "001-user-auth-jwt.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Projects API, Access (v1 — project owner + shared users)

## What to build

Project CRUD for owner; share/unshare collaborator by username; list only projects owned or shared with caller. Tests for isolation between users.

## Dev / test data

- [ ] Seed or fixture: sample project + owner (depends on user seed from 001).

## Acceptance criteria

- [ ] Owner can create, read, update, delete own projects (slug, name, guidelines)
- [ ] Owner can add/remove shared user by username; shared user sees project in list
- [ ] Non-member cannot read or mutate project (403/404)
- [ ] Tests cover owner CRUD, share access, and non-member denial

## Blocked by

- `issues/001-user-auth-jwt.md`

## User stories addressed

- 2
- 3
- 4
- 5
- 13
