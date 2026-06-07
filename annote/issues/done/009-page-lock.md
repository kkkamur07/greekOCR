---
id: "009"
title: "page-lock"
type: AFK
status: done
blocked_by: []
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Page lock (manual + 100% prompt), nested `PageLockSettings`, locked state on catalogue and editor

## What to build

End-to-end **Page lock** for one page: persist `locked` on the page annotation (or equivalent), expose lock/unlock via API, reject annotation mutations while locked (409 conflict), and update the editor + page list UI.

- **Lock** manually from the editor at any time.
- When **pairing progress** reaches 100%, show a dismissible prompt offering to lock; accepting locks the page.
- While locked: disable segment draw/edit/delete and pairing changes; show locked badge on page list and editor header.
- While locked: **Export** and live **Preview PDF** remain available.
- Add nested **`PageLockSettings`** (`prompt_at_full_pairing`, etc.) to application settings.

This slice does not include annotation history or frozen share PDF (slices 010–011).

## Acceptance criteria

- [x] `POST /pages/{stem}/lock` sets locked state; `POST /pages/{stem}/unlock` clears it.
- [x] `GET /pages` and `GET /pages/{stem}` expose `locked` (or equivalent) to the client.
- [x] `PUT /pages/{stem}/annotation` and auto-segment return **409** when the page is locked.
- [x] Editor canvas and pairing controls are disabled when locked; unlock re-enables them.
- [x] Lock button and locked indicator visible in editor and on home page cards.
- [x] At 100% pairing, lock prompt appears once per completion (dismissible); accept locks the page.
- [x] `PageLockSettings` is nested in `Settings` and overridable via env vars.
- [x] Backend integration tests cover lock, unlock, and rejected edit while locked.

## Blocked by

None — can start immediately.

## User stories addressed

- User story 55
- User story 56
- User story 57
- User story 58
- User story 59
- User story 60
- User story 61
- User story 75 (PageLockSettings portion)
