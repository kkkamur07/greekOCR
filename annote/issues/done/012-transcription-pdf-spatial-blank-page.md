---
id: "012"
title: "transcription-pdf-spatial-blank-page"
type: AFK
status: done
blocked_by: []
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Transcription PDF rules (spatial blank-page layout); supersedes facsimile overlay behaviour from slice 011.

## What to build

Replace the facsimile-plus-overlay **Transcription PDF** generator with a **spatial blank-page** PDF end-to-end:

- Single page matching manuscript image width and height; white background; **no** facsimile image drawn.
- **Paired segments only**: plain dark text placed horizontal inside each segment's **axis-aligned bounding box**, auto-sized to fit with word wrap.
- **No** highlight rectangles, borders, or shading behind text.
- When no segments are paired, still emit a valid blank PDF at the correct page dimensions.
- Live preview endpoint (`GET .../transcription.pdf`) returns PDF with **`Content-Disposition: inline`** so clients can embed without forcing download.
- Share PDF at lock uses the same generator (written to disk by existing share flow in slice 011).
- Greek Unicode rendering preserved via existing font registration.

See PRD **Transcription PDF rules** and user stories 48–53, 54 (layout portion).

## Acceptance criteria

- [x] Generated PDF page dimensions match the source page image (width × height).
- [x] PDF contains paired transcription text at segment positions; unpaired segments omitted.
- [x] PDF contains no embedded manuscript image (facsimile removed).
- [x] PDF has no text highlight boxes or borders — plain text only.
- [x] When zero segments are paired, PDF is a single blank page at correct dimensions.
- [x] Preview endpoint returns `application/pdf` with inline content disposition.
- [x] Greek Unicode paired text appears in extracted PDF text (existing font behaviour).
- [x] Backend tests cover paired-only text, blank unpaired page, and page dimensions.

## Blocked by

None — can start immediately.

## User stories addressed

- User story 48
- User story 49
- User story 50
- User story 51
- User story 52
- User story 53
