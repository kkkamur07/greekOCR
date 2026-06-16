---
id: "027"
title: "remove-root-app-duplicates"
type: AFK
status: review
blocked_by:
  - "019-authenticated-platform-shell.md"
  - "023-page-review-status.md"
  - "024-annotation-history-restore.md"
  - "025-export-approved-line-artifacts.md"
  - "026-transcription-pdf-artifact.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Remove duplicate root production app folders after merge.

## What to build

Once the merged annote production app provides the platform shell, manual annotation, review, history, export, and PDF workflows, remove or deprecate the duplicate root production app folders so there is one production app. Preserve the root model workspace and any root-level tests that genuinely belong to model or whole-repo integration concerns.

## Acceptance criteria

- [x] Duplicate root backend app code is removed or clearly deprecated after equivalent annote behavior passes tests.
- [x] Duplicate root frontend app code is removed or clearly deprecated after equivalent annote behavior passes build/tests.
- [x] Duplicate root infrastructure app assets are removed or clearly deprecated after annote infrastructure works.
- [x] The root model workspace remains in place.
- [x] Root local data contents are untouched.
- [x] Documentation and commands point to the annote production app root.
- [x] Tests/builds pass after cleanup.

## Blocked by

- `issues/019-authenticated-platform-shell.md`
- `issues/023-page-review-status.md`
- `issues/024-annotation-history-restore.md`
- `issues/025-export-approved-line-artifacts.md`
- `issues/026-transcription-pdf-artifact.md`

## User stories covered

- 10
- 11
- 12
- 49
