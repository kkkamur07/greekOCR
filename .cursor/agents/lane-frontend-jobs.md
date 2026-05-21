---
name: lane-frontend-jobs
description: AFK parallel lane I from issues/dag.md — issue 016 on feat/016-frontend-jobs. Jobs panel polling OpenAPI job types after 006 segment jobs exist.
---

You own **lane I — frontend jobs panel** (`issues/016-frontend-jobs-panel.md`).

## Rules

- **Single branch:** `feat/016-frontend-jobs`
- **Blocked until:** 004, 006, 012 done
- Poll `GET /jobs/{id}`; show status, sanitized error, segment/transcribe job types
- `npm run build` before Review

## Done

- Issue `status: review`, `branch: feat/016-frontend-jobs`
