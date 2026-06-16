---
id: "023"
title: "page-review-status"
type: AFK
status: backlog
blocked_by:
  - "022-page-transcription-pairing-progress.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Human review and Page-level Review status.

## What to build

Add a Page-level Review status workflow. A researcher can mark a Document part/Page as reviewed or unreviewed independently from Pairing progress, and the frontend displays the boolean as clear Reviewed or Unreviewed labels.

## Acceptance criteria

- [ ] A project member can mark a Document part/Page as reviewed.
- [ ] A project member can mark a Document part/Page as unreviewed.
- [ ] Review status remains independent of Pairing progress, including partially paired Pages.
- [ ] Editing Line geometry or Ground truth text does not automatically flip Review status.
- [ ] The frontend displays Reviewed and Unreviewed labels.
- [ ] Unauthorized users cannot change Review status.
- [ ] API and UI tests cover toggling, partial-progress reviewed Pages, and persistence after reload.

## Blocked by

- `issues/022-page-transcription-pairing-progress.md`

## User stories covered

- 31
- 32
- 33
- 34
- 35
- 36
