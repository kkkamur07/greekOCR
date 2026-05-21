---
id: "016"
title: "frontend-jobs-panel"
type: AFK
status: done
blocked_by:
  - "done/004-job-runner.md"
  - "done/012-nextjs-openapi-codegen.md"
  - "006-segment-job-kraken.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Job panel (poll status, toasts)

## What to build

Editor **Jobs panel**: list active/recent jobs for document or part; poll `GET /jobs/{id}` until terminal state; toast on failure with `Job.error`; actions to enqueue segment/transcribe when APIs exist (006/009).

## Error handling

- [ ] Display sanitized job error string only; log details in dev console optional.

## Dev / test data

- [ ] Dev: `ENABLE_TEST_JOB_ROUTES=true` allows noop test job from UI smoke; document in `backend/core/.env.example`.
- [ ] Mock job list in Storybook optional (not required v1).

## Acceptance criteria

- [ ] Enqueue shows job id; UI polls until done/failed
- [ ] Failed job shows user-visible error message
- [ ] Concurrent jobs show independent status rows
- [ ] Works with test noop job without GPU

## Blocked by

- `issues/done/004-job-runner.md`
- `issues/done/012-nextjs-openapi-codegen.md`
- `issues/006-segment-job-kraken.md`

## User stories addressed

- 24
- 25
- 42
