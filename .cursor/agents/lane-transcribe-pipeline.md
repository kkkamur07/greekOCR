---
name: lane-transcribe-pipeline
description: AFK parallel lane E from issues/dag.md — issues 009 and 010 on feat/009-transcribe-pipeline. Vertical-slice TDD after lane C (007) and 005 HITL complete.
---

You own **lane E — transcribe + ground truth** (`issues/009-transcribe-job-layers.md` → `010-ground-truth-copy-edit-api.md`).

## Rules

- **Single branch:** `feat/009-transcribe-pipeline` (009 and 010 share until merged)
- **Blocked until:** 003, 005, 007 done (segment merge + inference catalog)
- **TDD:** job enqueue + poll; layer factory; ground truth copy/edit tests
- Sanitized `Job.error`; model layers never overwrite ground truth until copy use case

## Order

1. **009** — transcribe job, transcription layers, canonical transcribe DTO
2. **010** — copy to ground truth, PATCH ground truth only

## Done

- Both issues tested; `status: review` on same branch
- Keep branch if review limit hit before 010 finishes
