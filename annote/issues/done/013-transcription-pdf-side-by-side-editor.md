---
id: "013"
title: "transcription-pdf-side-by-side-editor"
type: AFK
status: done
blocked_by:
  - "012-transcription-pdf-spatial-blank-page.md"
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Transcription PDF editor UX (side-by-side panel, unified menu, canvas interaction).

## What to build

Editor experience for **Transcription PDF** review beside the manuscript — end-to-end through UI and API consumption:

- **Side-by-side layout**: opening Transcription PDF shows a panel beside the canvas (~50/50 split), not over the manuscript and not in a new browser tab.
- **Unified control**: single **Transcription PDF** split button with dropdown for **Preview** (live) and **Share** (frozen, disabled until page locked); primary click toggles live preview panel.
- **Inline embed**: fetch PDF bytes client-side and display via embedded viewer (blob URL / object tag); opening preview must **not** auto-download a file.
- **Explicit download**: separate Download action only when the user requests a file.
- **Live refresh**: preview reloads after annotation saves (and when opening/switching modes).
- **Share mode**: when locked, user can switch panel to frozen share PDF; share tab disabled when unlocked.
- **Canvas interaction**: manuscript pan and zoom remain usable while the PDF panel is open; view refits when panel opens/closes or container resizes; pan allowed on locked pages.
- Copy updated to describe spatial blank-page layout (not facsimile).

Depends on slice 012 for meaningful blank-page PDF content. Share persistence and lock gating remain from slice 011.

See PRD **Transcription PDF rules** (editor UX) and user stories 54–63, 69, 71.

## Acceptance criteria

- [x] Clicking Transcription PDF opens a side panel beside the canvas without navigating away or overlaying the manuscript.
- [x] Opening the panel does not trigger an automatic file download.
- [x] PDF displays inline in the panel (embedded viewer with loading/error states).
- [x] Unified Transcription PDF menu offers Preview and Share; Share disabled when page unlocked.
- [x] Download is a separate explicit action.
- [x] Live preview updates after annotation is saved.
- [x] User can pan and zoom the manuscript while the PDF panel is open.
- [x] Locked pages: pan works; share mode viewable in panel when locked.
- [x] Frontend tests cover menu actions, panel embed, and share disabled state.

## Blocked by

- `issues/012-transcription-pdf-spatial-blank-page.md`

## User stories addressed

- User story 54
- User story 55
- User story 56
- User story 57
- User story 58
- User story 59
- User story 60
- User story 61
- User story 62
- User story 63
- User story 69
- User story 71
