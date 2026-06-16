---
id: "019"
title: "authenticated-platform-shell"
type: "AFK"
status: "done"
blocked_by:
  - "018-annote-production-root.md"
parent_prd: "issues/prd-annote-merge.md"
---



## Parent

`issues/prd-annote-merge.md` — Authenticated Project and Document hierarchy inside annote.

## What to build

Provide the authenticated annote platform shell: users can log in, register, view their Projects, create or open a Project, create or open a Document, and navigate to ordered Document parts. The shell should use the platform frontend stack while preserving annote's visual identity where the editor begins.

## Acceptance criteria

- [ ] A user can register, log in, and stay authenticated in the merged annote frontend.
- [ ] A project member can list, create, and open Projects.
- [ ] A project member can list, create, and open Documents under a Project.
- [ ] A project member can list ordered Document parts for a Document.
- [ ] Non-members cannot access Projects, Documents, or Document parts they do not own or share.
- [ ] OpenAPI-generated frontend types are used for the platform shell API calls.
- [ ] API and UI tests cover authenticated access and unauthorized rejection.

## Blocked by

- `issues/018-annote-production-root.md`

## User stories covered

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 48
