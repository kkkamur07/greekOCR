# annote — Product Requirements Document

## Problem Statement

Researchers preparing Greek manuscript data for line-level OCR (e.g. Calamari) have full-page scan images and page-level transcriptions with line breaks, but lack a simple local tool to manually outline each written line on the image, pair that outline with the correct line of text, and export rectified line images plus matching text files for training.

Today this work is fragmented: ad hoc image editors do not preserve transcription pairings; platforms like eScriptorium are heavy and hosted; automatic segmenters (Kraken) still need human correction on difficult folios. The researcher needs a **standalone, filesystem-based** annotator that runs on localhost, resumes work across sessions, and produces consistently named training artefacts without users, jobs, or a database.

Once a folio is fully segmented and paired, researchers also need confidence that work will not be accidentally overwritten, a way to roll back after mishaps (bad edits, accidental auto-segment replace), and reviewable artefacts (visual progress, export previews, transcription PDFs) before sharing results with colleagues who do not use annote.

Researchers reviewing transcription PDFs need to compare the spatial layout of paired text against the manuscript while editing, without the PDF overlapping the canvas or triggering unwanted downloads. The PDF itself should be a clean press-style page — text at segment positions on a blank page — not a facsimile photograph with text painted on top.

**Kraken** auto-segmentation speeds up line outlining but produces polygons that are often too large: they bleed into neighbouring lines and create **Segment overlap** (shared interior area between segments). Manually clipping neighbours after every edit is tedious. Researchers need **Segment refinement** that automatically shrinks each Kraken polygon inward to actual ink edges while respecting the original Kraken boundary as a maximum extent, then leaves segments hand-editable without further automatic constraints.

## Solution

Build **annote** as a local browser application under `annote/`: a **Next.js App Router** frontend and a **standalone FastAPI** backend that read and write files under `annote/data/`. The researcher opens a **Page** from a flat list, draws **Segments** (polygon or rotated rectangle) on the full-page image, manually **pairs** each segment to a **Text line** from the page transcription, and **exports** processed line images and text files when ready.

Annotation autosaves to JSON so sessions are resumable. **Export** is manual, shows a dirty indicator when outputs are stale, exports **paired segments only**, and warns about unpaired segments and unused text lines. **Processing** is a pluggable pipeline; v1 implements polygon-to-rectangle **rectify** only. Future steps (normalize height, binarize) plug in without changing annotation data.

**Pairing progress** is visible on the page list and in the editor so researchers know how far pairing has gone. **Paired segments** are visually distinct on the canvas. **Export preview** shows a rectified JPEG crop for a selected segment before export.

**Transcription PDF** is a single-page PDF the same size as the **Page** image: a **blank white page** with **plain paired transcription text** placed at each **Segment**'s axis-aligned bounding box (no manuscript facsimile, no highlight boxes or borders). Text is auto-sized to fit each box. Only paired segments appear; if none are paired, the PDF is still emitted as a blank page at the same dimensions. Two modes: live **preview** (regenerated from current annotation) and frozen **share** (written at **Page lock**, served from disk until unlock). In the editor, **Transcription PDF** opens in a **side-by-side panel** next to the manuscript canvas (not as an overlay and not as an automatic download). A single **Transcription PDF** control offers **Preview** and **Share** choices; **Download** is explicit. The manuscript canvas remains pannable and zoomable while the preview panel is open.

**Page lock** freezes all segment geometry and pairings when annotation work is complete (manual lock anytime, or prompt at 100% pairing). While locked, the canvas is read-only but export and PDF preview/download remain available. **Unlock** returns to normal editing.

**Annotation history** keeps rolling timed snapshots plus protected milestone snapshots so researchers can **restore** a prior annotation state after mishaps, without git or a database.

**Segment refinement** runs automatically (**Auto-refine**) immediately after **Kraken** auto-segmentation. Each Kraken segment stores its original boundary as the **Kraken ceiling** (maximum extent during refinement) and its editable `points` as the refined polygon snapped to **Ink edge signal** (grayscale luminance edges, e.g. Canny) inside a fixed **Refinement margin** of **4 px** inward from the ceiling. Refined polygons are **Contour simplification**-reduced for hand editing. If refinement fails for one segment, **Refinement fallback** keeps the unrefined Kraken polygon as `points` while still storing the ceiling. **Segment source** distinguishes **manual** draws from **Kraken** segments. Hand-edits are never clamped to the ceiling. An optional **Kraken ceiling overlay** (dashed outline, toggleable, off by default) shows the original Kraken boundary for the selected segment.

Primary outputs per paired segment: `<manuscript_name>_<segment_number>.jpg` and `<manuscript_name>_<segment_number>.txt`, where **manuscript name** is the page image filename stem and **segment number** is assigned in creation order.

## User Stories

### Setup and navigation

1. As a researcher, I want to start annote with a single documented command (or two), so that I can work locally without Docker or Postgres.
2. As a researcher, I want the tool to scan a pages directory for manuscript images, so that I do not manually register each file.
3. As a researcher, I want a simple sorted page list on the home screen, so that I can pick which folio to annotate.
4. As a researcher, I want to click a page and open an editor view, so that I can focus on one image at a time.
5. As a researcher, I want to return to the page list from the editor, so that I can switch folios without restarting the app.

### Page images and transcriptions

6. As a researcher, I want the editor to display the full page image with zoom and pan, so that I can work on large high-resolution scans.
7. As a researcher, I want each page image to have a matching page transcription file (same filename stem), so that text is available while I annotate.
8. As a researcher, I want the page transcription split into numbered text lines by line breaks, so that I can see how many lines the text expects.
9. As a researcher, I want to add page images and transcription files by dropping them into the data folders, so that I do not need an upload UI in v1.
10. As a researcher, I want a clear message when a page image has no transcription file, so that I know what is missing.

### Segment drawing

11. As a researcher, I want to draw a polygon segment around one written line, so that skewed or irregular lines are captured accurately.
12. As a researcher, I want to draw a rectangle segment using click-drag and corner placement, so that straight-ish lines are faster to outline than a full polygon.
13. As a researcher, I want rectangle segments to support rotation (not axis-aligned-only boxes), so that lines photographed at an angle are covered cleanly.
14. As a researcher, I want to switch between polygon and rectangle tools, so that I can use the right shape per line.
15. As a researcher, I want each new segment to receive a segment number in creation order, so that export filenames are stable and predictable.
16. As a researcher, I want to select a segment by clicking it on the canvas, so that I can review or pair it.
17. As a researcher, I want to move polygon/rectangle vertices after drawing, so that I can correct mistakes without redrawing.
18. As a researcher, I want to delete a segment, so that I can remove bad outlines.
19. As a researcher, I want undo-friendly editing (at minimum vertex adjust and delete), so that annotation mistakes are recoverable.
20. As a researcher, I want segment outlines visible over the image with a clear selected state, so that I can see what is already annotated.
21. As a researcher, I want paired segments to use a different highlight color than unpaired segments, so that I can see pairing status at a glance on the canvas.

### Persistence and sessions

22. As a researcher, I want segment geometry autosaved to disk, so that I do not lose work if I close the browser.
23. As a researcher, I want to reopen a page and see my previous segments, so that I can continue across sessions.
24. As a researcher, I want annotation data stored separately from exported training files, so that I can re-export after editing without losing source transcriptions.
25. As a researcher, I want the tool to tolerate a page with zero segments, so that I can open transcriptions before drawing.

### Pairing

26. As a researcher, I want pairing to start by selecting a segment first, so that the workflow matches how I look at the image.
27. As a researcher, I want to pick a text line from a list for the selected segment, so that I can match image lines to existing transcription without retyping.
28. As a researcher, I want to edit the paired text inline, so that I can fix typos in the transcription without leaving the tool.
29. As a researcher, I want to see which text lines are already paired, so that I do not assign the same line twice by mistake.
30. As a researcher, I want to see which segments are unpaired, so that I know what work remains before export.
31. As a researcher, I want pairing saved in the annotation file, so that pairings resume when I reopen the page.
32. As a researcher, I want no automatic text-to-segment matching, so that I stay in control of ambiguous layouts.
33. As a researcher, I want to see pairing progress (paired segments vs total segments) on the editor and page list, so that I know how complete the page is.
34. As a researcher, I want a segment to count as paired when it has a text-line link or inline text override, so that typos fixed inline still count as done.

### Export and processing

35. As a researcher, I want a manual Export action, so that processed files are written only when I am ready.
36. As a researcher, I want a dirty indicator when annotations or pairings changed since the last export, so that I know outputs may be stale.
37. As a researcher, I want export to write a processed JPEG per paired segment, so that I have training images on disk.
38. As a researcher, I want export to write a matching `.txt` file per paired segment, so that each image has ground-truth text alongside it.
39. As a researcher, I want exported files named `<page_stem>_<segment_number>`, so that outputs are distinguishable and sortable.
40. As a researcher, I want unpaired segments skipped on export, so that partial progress is allowed.
41. As a researcher, I want warnings listing unpaired segments and unused text lines on export, so that I notice incomplete pages.
42. As a researcher, I want polygon segments rectified to a rectangle image on export, so that line crops are axis-aligned for OCR training.
43. As a researcher, I want rectangle segments rectified consistently on export, so that all outputs go through the same pipeline.
44. As a researcher, I want processing implemented as pluggable steps, so that binarization or height normalization can be added later without re-annotating.
45. As a researcher, I want to re-export after editing segments or pairings, so that training files can be refreshed.
46. As a researcher, I want exported line images and text files in dedicated output directories, so that my source pages and transcriptions stay untouched.
47. As a researcher, I want to preview the rectified export crop for a selected segment before exporting the page, so that I can spot bad geometry early.

### Transcription PDF

48. As a researcher, I want a transcription PDF that is a blank page the same size as my manuscript image with paired text placed at segment positions, so that I can review the spatial layout of pairings without a facsimile photograph underneath.
49. As a researcher, I want only paired segments to appear in the transcription PDF, so that the layout reflects confirmed pairings only.
50. As a researcher, I want Greek Unicode to render correctly in the transcription PDF, so that polytonic text is legible.
51. As a researcher, I want transcription text auto-sized to fit each segment's bounding box, so that long lines remain readable without manual formatting.
52. As a researcher, I want plain text only in the PDF (no boxes, borders, or facsimile highlights), so that the output is a clean press-style page.
53. As a researcher, I want a blank transcription PDF at the correct page size when no segments are paired yet, so that preview still works while pairing is in progress.
54. As a researcher, I want a frozen share PDF written when I lock the page, so that colleagues receive a stable artefact that does not change if I unlock and edit later.
55. As a researcher, I want Share PDF available only when the page is locked, so that sharing implies annotation is intentionally frozen.
56. As a researcher, I want to re-download the share PDF after lock without regenerating it from live state, so that the on-disk share file remains canonical until unlock.
57. As a researcher, I want to open a live transcription PDF preview beside the manuscript in the editor, so that I can compare layout and pairings without leaving the annotation view.
58. As a researcher, I want opening the transcription PDF preview not to download a file automatically, so that review is uninterrupted.
59. As a researcher, I want a single Transcription PDF control with Preview and Share options, so that I do not hunt for separate buttons.
60. As a researcher, I want an explicit Download action when I need a file on disk, so that preview and download are separate intents.
61. As a researcher, I want the live preview to refresh after annotation saves, so that edits appear in the spatial layout without reopening the panel.
62. As a researcher, I want to pan and zoom the manuscript canvas while the transcription PDF panel is open, so that I can cross-check ink against the text layout.
63. As a researcher, I want to switch between live Preview and frozen Share inside the PDF panel when the page is locked, so that I can compare draft layout with the locked artefact.

### Page lock

64. As a researcher, I want to lock a page manually when I consider annotation complete, so that segment geometry and pairings cannot be changed accidentally.
65. As a researcher, I want to be prompted to lock when pairing reaches 100%, so that finishing pairing naturally leads to protecting the work.
66. As a researcher, I want to dismiss the 100% lock prompt and keep editing, so that I am not forced to lock before reviewing exports or previews.
67. As a researcher, I want the canvas and pairing controls disabled while a page is locked, so that locked state is obvious and enforced.
68. As a researcher, I want to unlock a page when I need to make corrections, so that lock is not permanent.
69. As a researcher, I want export and transcription PDF preview/download to remain available on a locked page, so that I can still produce deliverables from frozen annotation.
70. As a researcher, I want the page list to show whether a page is locked, so that I can see completion status across folios.
71. As a researcher, I want to pan the manuscript on a locked page, so that I can still navigate the image while editing is frozen.

### Annotation history

72. As a researcher, I want annotation snapshots saved periodically while I edit, so that I can recover from recent mistakes.
73. As a researcher, I want snapshots at 50% and 100% pairing progress, so that milestone states are always recoverable.
74. As a researcher, I want snapshots when I lock or unlock a page, so that state changes around completion are preserved.
75. As a researcher, I want only the last five timed snapshots kept per page, so that disk use stays bounded.
76. As a researcher, I want milestone snapshots (50%, 100%, lock, unlock) kept even when the timed cap is reached, so that important checkpoints are never evicted.
77. As a researcher, I want to list available history snapshots for a page with human-readable labels, so that I can pick the right restore point.
78. As a researcher, I want to restore a prior snapshot, so that I can undo catastrophic edits (e.g. accidental auto-segment replace).
79. As a researcher, I want restore to replace the current annotation, so that the canvas reflects the chosen snapshot immediately.
80. As a researcher, I want unlock to remain available after restore, so that I can edit from a restored baseline.

### Kraken auto-segmentation and segment refinement

86. As a researcher, I want Kraken auto-segmentation to produce tighter line polygons that hug ink edges, so that neighbouring segments overlap less and need less manual correction.
87. As a researcher, I want refinement to run automatically right after Kraken segmentation, so that I see improved segments immediately without an extra button.
88. As a researcher, I want refined segments to shrink inward only and never grow beyond the original Kraken boundary, so that I trust auto-refinement not to steal ink from neighbours.
89. As a researcher, I want the original Kraken boundary stored per segment, so that I can see how far refinement moved the outline.
90. As a researcher, I want segments that fail refinement to keep the unrefined Kraken polygon, so that one bad line does not block the rest of the page.
91. As a researcher, I want refined polygons simplified to fewer vertices, so that I can adjust corners by hand without fighting dense Kraken point runs.
92. As a researcher, I want to edit refined segment vertices freely without automatic clamping to the Kraken ceiling, so that I can fix stubborn lines manually.
93. As a researcher, I want manually drawn segments distinguished from Kraken-generated ones, so that I know which outlines came from auto-segmentation.
94. As a researcher, I want to toggle a dashed Kraken ceiling outline for the selected segment, so that I can compare refined shape against the original Kraken box when needed.
95. As a researcher, I want the Kraken ceiling overlay off by default, so that the canvas stays uncluttered during normal editing.
96. As a researcher, I want segment interiors not to overlap after refinement where possible, so that each line claims distinct ink area (shared edges and corners are fine).
97. As a researcher, I want refinement to use ink-edge detection on the page image inside each Kraken region, so that outlines follow visible stroke boundaries.
98. As a researcher, I want auto-segment to still pair segments to transcription lines by reading order when a page transcription exists, so that refinement does not change the pairing workflow.
99. As a researcher, I want append-mode Kraken segmentation to refine only the newly added segments, so that existing manual work is preserved.
100. As a researcher, I want replace-mode Kraken segmentation to replace all segments with refined Kraken polygons, so that a full re-segment gets a clean slate.
101. As a researcher, I want locked pages to reject Kraken auto-segment, so that frozen annotation cannot be overwritten accidentally.
102. As a developer, I want segment refinement testable on synthetic ink images without the UI, so that contour behaviour can be verified in isolation.
103. As a developer, I want backward-compatible annotation JSON for pages annotated before refinement shipped, so that existing work loads without migration scripts.

### Developer and quality

81. As a developer, I want OpenAPI-generated types from the FastAPI schema, so that the Next.js client stays aligned with the API.
82. As a developer, I want the processing pipeline testable without the UI, so that rectify logic can be verified in isolation.
83. As a developer, I want the filesystem data layout documented, so that I know where to place inputs and find outputs.
84. As a researcher, I want Greek Unicode text handled correctly in transcriptions and exports, so that polytonic Greek is not corrupted.
85. As a developer, I want lock, history, and PDF policy configurable via nested Pydantic settings classes, so that intervals and retention can be tuned without code changes.

## Implementation Decisions

### Architecture

- **Standalone annote app** in `annote/frontend` (Next.js App Router) and `annote/backend` (FastAPI). Not coupled to the Kalamos platform API, Postgres, users, or jobs.
- **Filesystem persistence** under `annote/data/` with no database.
- **Build order**: v1 slices (UI, persistence, export) delivered; phase 2 adds pairing progress UI, export preview, transcription PDF, page lock, annotation history, transcription PDF share mode; phase 3 refines transcription PDF to spatial blank-page layout and side-by-side editor preview.
- Canvas drawing logic may be adapted from Kalamos legacy `ImageCanvas` concepts, but the annote shell is greenfield Next.js.

### Deep modules (testable interfaces)

| Module | Responsibility | Interface sketch |
|--------|----------------|------------------|
| **Data layout** | Resolve paths for pages, transcriptions, annotations, exports, history, share PDFs | `resolve_page_paths(stem)`, `list_pages()` |
| **Page catalogue** | Discover page images, transcriptions, lock state, pairing progress | `list_pages() -> PageSummary[]`, `build_page_summary(stem)` |
| **Text line parser** | Split page transcription into ordered text lines | `split_text_lines(text) -> TextLine[]` |
| **Segment text** | Resolve exportable text for a segment; pairing progress helpers | `segment_text(segment, lines)`, `compute_pairing_progress(...)` |
| **Annotation store** | Load/save segment geometry, pairings, lock flag, export metadata | `load_annotation(stem)`, `save_annotation(stem, data)` |
| **Page lock service** | Lock/unlock page; enforce read-only when locked | `lock_page(stem)`, `unlock_page(stem)`, `assert_editable(stem)` |
| **History service** | Write/list/prune/restore annotation snapshots | `maybe_snapshot(stem, reason)`, `list_history(stem)`, `restore_history(stem, id)` |
| **Export state** | Compare annotation revision vs last export timestamp/hash | `is_export_dirty(stem)`, `mark_exported(stem)` |
| **Processing pipeline** | Run ordered steps on a segment crop | `process(image, segment, steps) -> Image` |
| **Export service** | Export paired segments, collect warnings | `export_page(stem) -> ExportResult` |
| **Preview service** | JPEG preview of rectified segment crop | `preview_segment_jpeg(stem, segment_id) -> bytes` |
| **Transcription PDF** | Spatial blank-page PDF and frozen share PDF at lock | `generate_transcription_pdf(stem) -> bytes`, `write_share_pdf(stem)`, `read_share_pdf(stem)` |
| **App settings** | Nested Pydantic config for history, lock, PDF policy | `HistorySettings`, `PageLockSettings`, `TranscriptionPdfSettings` composed in `Settings` |
| **Page API** | HTTP layer over catalogue, annotations, lock, history, PDF | REST routes |
| **Canvas editor** | Draw/select/edit segments; respect lock; pan while PDF panel open | Client component |
| **Transcription PDF panel** | Side-by-side preview/share embed, explicit download, refresh on save | Client component |
| **Pairing panel** | Text line list, inline edit, export preview | Client component |
| **Kraken segment service** | Run BLLA; convert lines to segments; optional transcription pairing | `segment_image(image)`, `auto_segment_page(stem, ...)` |
| **Segment refinement** | Active-contour shrink inside Kraken ceiling; simplification; per-segment fallback | `refine_segment(image, ceiling, *, margin_px=4) -> RefineResult` |
| **Segment source** | Distinguish manual vs Kraken provenance on each segment | `source: "manual" \| "kraken"` on `Segment` |

### Segment refinement

- **Trigger**: **Auto-refine** runs inside the Kraken auto-segment pipeline immediately after Kraken polygons are produced, before persistence and API response.
- **Kraken ceiling**: For each Kraken segment, store the merged Kraken polygon (pre-refinement) in `kraken_ceiling`. Set `points` to the refined polygon (or unrefined Kraken polygon on **Refinement fallback**).
- **Refinement margin**: Fixed **4 px** inward from the ceiling in v1 (configurable constant; adaptive per-line height deferred).
- **Ink edge signal**: Grayscale luminance → edge map (e.g. Canny) on the page image crop masked to the ceiling polygon.
- **Algorithm**: Active contour / snake constrained to stay inside the ceiling and at least `margin_px` inset from it; snap to strong edges.
- **Contour simplification**: Douglas–Peucker (or equivalent) with ~2 px tolerance after refinement to reduce vertex count.
- **Refinement fallback**: Per-segment — on failure (no stable contour, degenerate result, contour escapes ceiling), `points` = unrefined Kraken polygon; `kraken_ceiling` still stored; other segments on the page continue.
- **Hand-edits**: Vertex moves and new manual segments are never clamped to `kraken_ceiling`.
- **Segment source**: `manual` for user-drawn segments (`kraken_ceiling` null); `kraken` for auto-segment output. Defaults to `manual` when field absent (backward compatibility).
- **Kraken ceiling overlay**: Editor-only dashed outline of `kraken_ceiling` for the selected segment; toggle in toolbar; **off by default**; not shown for manual segments.
- **Visual source distinction**: Subtle canvas styling difference between `manual` and `kraken` segments (e.g. stroke dash or colour); exact styling agent-chosen.
- **Overlap repair** (manual neighbour clip) remains **deferred** — refinement is the primary strategy to reduce **Segment overlap** from fat Kraken boxes.

### On-disk layout

- `data/manuscripts/pages/` — source page images
- `data/transcriptions/pages/` — page transcription text files (matching stems)
- `data/annotations/pages/` — per-page JSON (segments, pairings, lock flag, export metadata)
- `data/annotations/history/<stem>/` — annotation history snapshots (JSON + metadata per snapshot)
- `data/manuscripts/share/` — frozen share transcription PDFs (`<stem>_transcription.pdf`), written at lock
- `data/manuscripts/export/` — exported processed JPEGs and line `.txt` files (`<stem>_<segment_number>.{jpg,txt}`)

### API contracts (conceptual)

- `GET /pages` — list pages with stem, has_transcription, segment_count, export_dirty, pairing progress, locked
- `GET /pages/{stem}` — page detail + annotation summary
- `GET /pages/{stem}/image` — serve page image bytes
- `GET /pages/{stem}/transcription` — page transcription text + parsed text lines
- `GET /pages/{stem}/annotation` — full annotation JSON (includes locked flag)
- `PUT /pages/{stem}/annotation` — replace annotation (rejected with 409 when locked)
- `POST /pages/{stem}/segment` — Kraken BLLA auto-segment; runs **Auto-refine** before save; returns segments with `source`, `kraken_ceiling`, refined `points` (rejected when locked)
- `POST /pages/{stem}/export` — run processing + write line outputs; return warnings
- `GET /pages/{stem}/segments/{segment_id}/preview` — rectified JPEG preview for one segment
- `GET /pages/{stem}/transcription.pdf` — live preview transcription PDF (`Content-Disposition: inline` for embedding)
- `GET /pages/{stem}/transcription.share.pdf` — frozen share PDF (`Content-Disposition: attachment` for download; embeddable via client fetch)
- `POST /pages/{stem}/lock` — lock page; write share PDF; record history snapshot
- `POST /pages/{stem}/unlock` — unlock page; invalidate share PDF; record history snapshot
- `GET /pages/{stem}/history` — list snapshots with id, timestamp, reason, pairing progress at snapshot
- `POST /pages/{stem}/history/{snapshot_id}/restore` — restore snapshot (rejected when locked unless policy allows unlock-first)

### Page lock rules

- Lock freezes segment geometry and all pairings (full `PageAnnotation` content except lock metadata).
- Manual lock anytime; prompt at 100% pairing (dismissible).
- Unlock returns to editable state.
- Mutating routes (annotation save, auto-segment, segment draw persistence) return conflict when locked.
- Export and transcription PDF preview/download remain allowed on locked pages.
- Pan and zoom remain allowed on locked pages; draw/edit/pairing do not.

### Annotation history rules

- Timed snapshot: every five minutes during active editing on a page (configurable).
- Milestone snapshots: pairing progress crosses 50% or 100% (first crossing per direction per session or per page — implementation may dedupe repeat crossings).
- Event snapshots: lock and unlock always write protected snapshots.
- Retention: five rolling timed snapshots; milestone and lock/unlock snapshots are protected from eviction.
- Restore replaces current annotation on disk and in editor; does not delete history entries.

### Transcription PDF rules

- **Layout**: single page matching manuscript image width and height; white background; no facsimile image.
- **Text placement**: paired segments only; horizontal plain text inside each segment's axis-aligned bounding box; auto-fit font size with word wrap; dark text on white (not configurable in v1).
- **Empty pairing**: emit blank page at correct dimensions when no segments are paired.
- **Preview PDF**: regenerated on each request from current saved annotation; served with inline disposition for embedding.
- **Share PDF**: written once at lock from annotation at lock time; served from disk until page unlocked; same spatial blank-page layout as preview.
- **Editor UX**: side-by-side panel (50/50 split); fetch PDF as blob and embed (no automatic download on open); unified Transcription PDF menu with Preview and Share entries; explicit Download button; preview refreshes after annotation save; canvas refits on panel open/resize so pan/zoom keep working.

### Configuration (Pydantic)

Nested settings classes on application `Settings`, env-configurable (e.g. `ANNOTE_HISTORY__SNAPSHOT_INTERVAL_MINUTES=5`):

- **HistorySettings** — `snapshot_interval_minutes` (default 5), `max_timed_snapshots` (default 5), `pairing_milestones` (default `[50, 100]`)
- **PageLockSettings** — `prompt_at_full_pairing` (default true)
- **TranscriptionPdfSettings** — share output directory name, filename pattern

### Geometry

- Store segment coordinates in **natural image pixel space**.
- Polygon: ordered list of `[x, y]` points (four or more).
- Rectangle: four corners as ordered points.
- Transcription PDF uses axis-aligned bounding box of segment points for text placement (not rotated to polygon angle).
- **Segment** schema extensions: `source` (`manual` \| `kraken`, default `manual`), `kraken_ceiling` (`list[list[float]]` \| `null`, null for manual segments).

### Export rules

- Manual trigger only; UI shows dirty state when annotation is newer than last successful export.
- Export paired segments only; skip unpaired.
- Warn with counts/lists of unpaired segments and unused text lines.
- Segment number in filenames = creation order at segment creation (stable segment id in JSON).

### Session model

- Resumable: autosaved JSON reloads segments and pairings.
- Export is the checkpoint for training deliverables.
- Page lock is the checkpoint for frozen annotation and share PDF.
- Annotation history provides rollback without git.

## Testing Decisions

**What makes a good test**: Assert externally visible behavior — API status codes, files on disk, PDF bytes contain expected Greek text and omit facsimile imagery, PDF page dimensions match source image, locked pages reject edits but allow pan, preview panel embeds without navigation — not internal timer implementation or React state details.

| Module | Test focus |
|--------|------------|
| **Segment text / pairing progress** | Paired count, text override counts as paired, unused line count |
| **Page lock service** | Lock/unlock round-trip; PUT annotation 409 when locked; share PDF written at lock |
| **History service** | Timed retention cap; milestones protected; restore replaces annotation |
| **Transcription PDF** | Blank spatial layout; paired text only; blank when unpaired; page dimensions; share PDF stable after unlock + edit |
| **Preview service** | JPEG bytes for paired and unpaired segments |
| **Export state** | Dirty after edit; clean after export |
| **Export service** | Paired-only files; warnings |
| **Page API** | Integration tests with temp data directory; preview endpoint inline disposition |
| **Transcription PDF panel (UI)** | Menu opens preview mode; share disabled when unlocked; embed loads via fetch; download is explicit |
| **Segment refinement service** | Refined polygon inside ceiling; respects margin; simplified vertex count; fallback returns ceiling as points; synthetic ink fixtures |
| **Kraken auto-segment + refine** | POST `/segment` returns `source=kraken`, `kraken_ceiling` populated; refined `points` differ from ceiling on synthetic ink; append/replace modes |
| **Kraken ceiling overlay (UI)** | Toggle shows/hides dashed ceiling for selected Kraken segment only; off by default |

**Test seams** (highest first):

1. **Segment refinement service** — pure function tests on synthetic page crops: refined polygon ⊆ ceiling, inset ≥ margin where possible, closer to synthetic ink than ceiling, fallback on blank crop.
2. **Kraken auto-segment pipeline** — `auto_segment_page` with monkeypatched Kraken lines: schema fields, per-segment refine/fallback, pairing preserved.
3. **POST `/segment` API** — integration test through HTTP (existing `test_kraken_segment` pattern).
4. **Editor overlay toggle** — Vitest on overlay component / canvas props for dashed ceiling visibility.

**Prior art**: `annote/backend/tests/` pytest fixtures with temp `data_root`; `test_kraken_segment.py`, `test_polygon.py`; frontend Vitest for pure helpers and component behavior.

**UI**: Manual QA for side-by-side layout, pan while panel open, and lock/share switching; automated tests for menu/panel behavior and disabled share when unlocked.

## Out of Scope

- Kalamos platform integration, Postgres, user authentication, JWT, jobs/workers
- Automatic text-to-segment alignment or OCR inference (beyond optional Kraken assist and **Auto-refine** already present)
- **Overlap repair** (manual subtract overlap from neighbour segment) — discussed, deferred in favour of **Segment refinement**
- Adaptive **Refinement margin** per line height (fixed 4 px in v1)
- Re-refining segments on demand via a separate UI button (v1 refines only at Kraken auto-segment time)
- Clamping hand-edits to **Kraken ceiling**
- Manuscript grouping UI, project/dashboard hierarchy, multi-user collaboration
- Block-level layout (columns, regions above line level)
- Packaging as Electron/desktop app; cloud deployment
- Import/export to eScriptorium XML or TEI (deferred; Kalamos compatibility later)
- Processing steps beyond `rectify` in v1 (normalize_height, binarize planned as pluggable follow-ups)
- Upload UI for images (filesystem drop only in v1)
- Segment number reordering UI; reading-order enforcement
- Git-like branching, merge, or full version control of annotations
- Marking unpaired segments in transcription PDF (deferred; paired-only layout for v1)
- Requiring 100% pairing before manual lock or before share PDF
- Multi-page manuscript PDF bundles (single page per PDF for v1)
- Facsimile-plus-overlay transcription PDF (replaced by spatial blank-page layout)
- Rotated text following polygon skew in transcription PDF (axis-aligned boxes only in v1)
- Configurable PDF colours or fonts in v1 (white page, dark text, existing Unicode font)
- PDF preview as full-screen modal over the manuscript canvas

## Further Notes

- Domain glossary: `annote/CONTEXT.md` (includes **Segment refinement**, **Kraken ceiling**, **Segment source**, **Segment overlap**, **Refinement fallback**).
- Delivered in issues 001–013: foundation through transcription PDF side-by-side editor.
- Remaining vertical slices: **Segment refinement** auto-refine pipeline (014), Kraken ceiling overlay and source styling (015).
- Rectangle draw gesture (exact click sequence) is agent-chosen in slice 004; iterate on review if needed.
- Primary user deliverables remain processed `.jpg` + `.txt` per paired segment; share transcription PDF is a secondary shareable artefact for colleagues who do not use annote.
