---
id: "014"
title: "segment-refinement-auto-refine"
type: AFK
status: ready
blocked_by: []
parent_prd: "issues/prd.md"
---

## Parent

`issues/prd.md` — **Segment refinement** / **Auto-refine** after Kraken segmentation.

## What to build

End-to-end **Segment refinement** wired into the Kraken auto-segment pipeline:

- Extend **Segment** schema with `source` (`manual` | `kraken`, default `manual`) and `kraken_ceiling` (`list[list[float]]` | `null`).
- Manual draws and legacy annotations without these fields behave as `source=manual`, `kraken_ceiling=null`.
- New **Segment refinement** module: given a page image crop and a Kraken ceiling polygon, shrink inward using active contours snapped to **Ink edge signal** (grayscale luminance edges, e.g. Canny) with a fixed **Refinement margin** of **4 px**. Apply **Contour simplification** (~2 px Douglas–Peucker tolerance). **Refinement fallback**: if refinement fails for one segment, set `points` to the unrefined Kraken polygon and continue other segments.
- **Auto-refine** runs inside `auto_segment_page` immediately after Kraken polygons are produced: set `source=kraken`, `kraken_ceiling` = merged Kraken boundary, `points` = refined polygon (or fallback).
- Regenerate OpenAPI / frontend types. Editor displays returned `points` with no new UI controls in this slice.
- Locked pages continue to reject `POST /segment`.

## Acceptance criteria

- [ ] `Segment` schema includes `source` and `kraken_ceiling`; backward-compatible defaults for existing JSON
- [ ] `POST /segment` (replace and append modes) returns Kraken segments with `source=kraken` and populated `kraken_ceiling`
- [ ] Refined `points` are inside `kraken_ceiling` on synthetic ink test fixtures; vertex count reduced vs raw Kraken boundary
- [ ] Per-segment **Refinement fallback** keeps unrefined Kraken polygon when refinement cannot find a stable contour
- [ ] Manual segments created in the editor persist with `source=manual` and `kraken_ceiling=null`
- [ ] Transcription pairing by reading order still works after auto-segment + refine
- [ ] Unit tests for refinement service; integration tests for auto-segment pipeline (monkeypatched Kraken acceptable)

## Blocked by

None — can start immediately.

## User stories addressed

- 86
- 87
- 88
- 89
- 90
- 91
- 92
- 97
- 98
- 99
- 100
- 101
- 102
- 103
