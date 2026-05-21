---
id: "002"
title: "projects-sharing"
type: AFK
status: review
branch: feat/002-projects-sharing
blocked_by:
  - "001-user-auth-jwt.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Projects API, Access (v1 — project owner + shared users)

## What to build

Project CRUD for owner; share/unshare collaborator by username; list only projects owned or shared with caller. Tests for isolation between users.

## Error handling

- [x] Raise `NotFoundError` / `AccessDeniedError` from project services; rely on global handlers from 001.

## Dev / test data

- [x] Seed or fixture: sample project + owner (depends on user seed from 001).

## Acceptance criteria

- [x] Owner can create, read, update, delete own projects (slug, name, guidelines)
- [x] Owner can add/remove shared user by username; shared user sees project in list
- [x] Non-member cannot read or mutate project (403/404)
- [x] Tests cover owner CRUD, share access, and non-member denial

## Blocked by

- `issues/001-user-auth-jwt.md`

## User stories addressed

- 2
- 3
- 4
- 5
- 13
