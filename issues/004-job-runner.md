---
id: "004"
title: "job-runner"
type: AFK
status: backlog
blocked_by:
  - "done/000-platform-foundation.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Job runner module, Job execution

## What to build

Postgres-backed Job table with enqueue, transactional claim (`SKIP LOCKED` or equivalent), status transitions (`pending` → `running` → `done` | `failed`), and `GET /jobs/{id}`. Background worker loop runnable inside API process (compatible with multi-worker claim). Include a noop/sleep test handler to verify pipeline without GPU.

## Dev / test data

- [ ] Reuse seeded user/project/document from 001–003 where job FKs are needed; noop job tests may omit document FK.

## Acceptance criteria

- [ ] `POST` test enqueue returns `job_id` immediately; poll until `done`
- [ ] Two concurrent claimers do not execute the same job twice (test)
- [ ] Failed handler stores error message on job row
- [ ] Job record stores type, payload, timestamps, optional user/document/part FKs

## Blocked by

- `issues/000-platform-foundation.md`

## User stories addressed

- 24
- 25
- 40
- 41
- 42
