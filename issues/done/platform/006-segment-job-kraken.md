---
id: "006"
title: "segment-job-kraken"
type: "AFK"
status: "done"
blocked_by:
  - "done/003-documents-parts-media.md"
  - "done/004-job-runner.md"
  - "005-inference-catalog-bindings.md"
parent_prd: "issues/prd.md"
---



## Parent PRD

`issues/prd.md` — Segment job, Kraken adapter, canonical segment DTO

## What to build

`POST` segment on a DocumentPart → enqueue **Job**; worker invokes Kraken **ModelAdapter** → **CanonicalSegmentResult** (blocks + lines). Persist raw result on job; call **Segment merge** module (007) or inline minimal persist for first slice if 007 is split. HTTP returns `{ job_id }` immediately; client polls `GET /jobs/{id}`.

Adapter lives under `inference` / `ocr/` without importing FastAPI.

## Error handling

- [ ] Job failures store sanitized `Job.error`; missing part/model → `NotFoundError` at enqueue time.

## Dev / test data

- [ ] Reuse seeded project/document/part from 003 + inference seed from 005.
- [ ] Optional `ENABLE_SEGMENT_SMOKE=1` test route or pytest marker using tiny fixture image (skip in CI without GPU if documented).

## Acceptance criteria

- [ ] Authenticated project member can enqueue segment for a part
- [ ] Job completes with `done` and result JSON referencing blocks/lines count
- [ ] Non-member cannot enqueue (403/404)
- [ ] Tests: enqueue + poll noop/small fixture OR mocked adapter

## Blocked by

- `issues/done/003-documents-parts-media.md`
- `issues/done/004-job-runner.md`
- `issues/005-inference-catalog-bindings.md`

## User stories addressed

- 16
- 17
- 24
- 25
- 39
- 43
