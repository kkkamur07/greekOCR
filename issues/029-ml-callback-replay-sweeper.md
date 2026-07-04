---
id: "029"
title: "ml-callback-replay-sweeper"
type: AFK
status: backlog
blocked_by:
  - "028-ocr-prediction-execution-design.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — ML job execution and product job status integration.

## Problem

The v1 ML worker posts terminal job callbacks with bounded in-process retries. If every callback attempt fails, the ML job is already terminal in `ml_jobs`, but the platform Product job remains `running`/waiting because no sweeper or replay path reconciles terminal ML jobs back into the platform job table.

## What to build

Add a durable callback recovery path before relying on ML callbacks in production flows. Acceptable designs include a replay endpoint/command, a platform-side sweeper that reconciles terminal ML jobs, an ML-side undelivered-callback queue, or an explicit dead-letter state visible to operators.

## Acceptance criteria

- [ ] Exhausted callback delivery cannot leave a platform job waiting forever without operator-visible state.
- [ ] Recovery is durable across ML worker process restarts.
- [ ] Replays are idempotent for already-terminal platform jobs.
- [ ] Tests cover callback exhaustion followed by successful replay or sweep.
- [ ] Runbook notes how to detect and recover orphaned platform jobs.

## Blocked by

- `issues/028-ocr-prediction-execution-design.md`
