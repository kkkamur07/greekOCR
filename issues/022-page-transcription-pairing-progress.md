---
id: "022"
title: "page-transcription-pairing-progress"
type: AFK
status: review
blocked_by:
  - "021-editor-page-line-geometry.md"
parent_prd: "issues/prd-annote-merge.md"
---

## Parent

`issues/prd-annote-merge.md` — Page transcription helper, Pairing, Ground truth Line transcription, and Pairing progress.

## What to build

Add the manual Pairing workflow to the merged editor. A researcher can import or paste partial Page transcription text, split it into candidate Text lines, select a Segment first, pair or type approved text for that Line, and see Pairing progress based on paired Lines versus total Lines.

## Acceptance criteria

- [ ] A project member can import or paste partial Page transcription text for a Document part/Page.
- [ ] Imported Page transcription text is split into candidate Text lines without becoming canonical Ground truth automatically.
- [ ] A user can select a Segment/Line and pair it with a candidate Text line.
- [ ] A user can type or edit approved text directly for a selected Segment/Line.
- [ ] Approved text persists as Ground truth Line transcription.
- [ ] Pairing progress updates from paired Lines divided by total Lines.
- [ ] Unused Text lines and unpaired Segments remain visible enough to guide the user.
- [ ] API and UI tests cover partial imports, pairing, direct text edits, reload, and Pairing progress.

## Blocked by

- `issues/021-editor-page-line-geometry.md`

## User stories covered

- 20
- 21
- 22
- 23
- 24
- 25
- 30
- 31
