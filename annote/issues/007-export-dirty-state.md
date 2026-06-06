---
id: "007"
title: "export-dirty-state"
type: AFK
status: backlog
blocked_by:
  - "006-segment-text-pairing.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Export dirty indicator and export metadata

## What to build

Track **Export state** in annotation metadata: record timestamp (or content hash) of last successful export; mark **dirty** when segments or pairings change afterward. Show dirty indicator on editor and page list. Stub `POST /pages/{stem}/export` that validates request and updates export metadata only (no image processing yet) OR returns `501` with structured response — enough to wire UI Export button and prove dirty/clean transitions.

Full file export lands in slice 008.

## Error handling

- [ ] Page never exported shows dirty by default.
- [ ] Export stub failure does not mark clean.

## Dev / test data

- [ ] Unit tests for export state: clean after mock export, dirty after annotation PUT.

## Acceptance criteria

- [ ] Editing segment or pairing sets export dirty
- [ ] Successful export stub clears dirty
- [ ] Page list shows dirty badge per page
- [ ] Editor shows dirty indicator and Export button
- [ ] Export state module tests pass

## Blocked by

- `issues/006-segment-text-pairing.md`

## User stories addressed

- 32
- 33
