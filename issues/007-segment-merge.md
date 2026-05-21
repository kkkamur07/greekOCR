---
id: "007"
title: "segment-merge"
type: AFK
status: backlog
blocked_by:
  - "006-segment-job-kraken.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Segment merge module (manual geometry preserved)

## What to build

**Segment merge** application service: `apply(part_id, canonical_segment, job_id)` — manual blocks/lines untouched; update/add/prune machine geometry; prune cascades **LineTranscription** rows for removed lines. Unit/integration tests are the primary deliverable (high priority per PRD).

## Error handling

- [ ] Invalid canonical DTO → `ValidationError`; missing part → `NotFoundError`.

## Dev / test data

- [ ] Pytest factories: part with one manual line + two machine lines; canonical payload samples in `tests/fixtures/segment_canonical.json`.
- [ ] No separate seed script required if factories build ORM rows in tests.

## Acceptance criteria

- [ ] Re-segment does not change `manual_geometry=true` lines/blocks
- [ ] New machine lines added; obsolete machine lines pruned
- [ ] Pruned line deletes all `LineTranscription` for that line
- [ ] Tests cover manual preservation, add, prune, and cascade delete

## Blocked by

- `issues/006-segment-job-kraken.md`

## User stories addressed

- 20
- 21
- 22
