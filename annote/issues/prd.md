# annote — Product Requirements Document

## Problem Statement

Researchers preparing Greek manuscript data for line-level OCR (e.g. Calamari) have full-page scan images and page-level transcriptions with line breaks, but lack a simple local tool to manually outline each written line on the image, pair that outline with the correct line of text, and export rectified line images plus matching text files for training.

Today this work is fragmented: ad hoc image editors do not preserve transcription pairings; platforms like eScriptorium are heavy and hosted; automatic segmenters (Kraken) still need human correction on difficult folios. The researcher needs a **standalone, filesystem-based** annotator that runs on localhost, resumes work across sessions, and produces consistently named training artefacts without users, jobs, or a database.

## Solution

Build **annote** as a local browser application under `annote/`: a **Next.js App Router** frontend and a **standalone FastAPI** backend that read and write files under `annote/data/`. The researcher opens a **Page** from a flat list, draws **Segments** (polygon or rotated rectangle) on the full-page image, manually **pairs** each segment to a **Text line** from the page transcription, and **exports** processed line images and text files when ready.

Annotation autosaves to JSON so sessions are resumable. **Export** is manual, shows a dirty indicator when outputs are stale, exports **paired segments only**, and warns about unpaired segments and unused text lines. **Processing** is a pluggable pipeline; v1 implements polygon-to-rectangle **rectify** only. Future steps (normalize height, binarize) plug in without changing annotation data.

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

### Persistence and sessions

21. As a researcher, I want segment geometry autosaved to disk, so that I do not lose work if I close the browser.
22. As a researcher, I want to reopen a page and see my previous segments, so that I can continue across sessions.
23. As a researcher, I want annotation data stored separately from exported training files, so that I can re-export after editing without losing source transcriptions.
24. As a researcher, I want the tool to tolerate a page with zero segments, so that I can open transcriptions before drawing.

### Pairing

25. As a researcher, I want pairing to start by selecting a segment first, so that the workflow matches how I look at the image.
26. As a researcher, I want to pick a text line from a list for the selected segment, so that I can match image lines to existing transcription without retyping.
27. As a researcher, I want to edit the paired text inline, so that I can fix typos in the transcription without leaving the tool.
28. As a researcher, I want to see which text lines are already paired, so that I do not assign the same line twice by mistake.
29. As a researcher, I want to see which segments are unpaired, so that I know what work remains before export.
30. As a researcher, I want pairing saved in the annotation file, so that pairings resume when I reopen the page.
31. As a researcher, I want no automatic text-to-segment matching, so that I stay in control of ambiguous layouts.

### Export and processing

32. As a researcher, I want a manual Export action, so that processed files are written only when I am ready.
33. As a researcher, I want a dirty indicator when annotations or pairings changed since the last export, so that I know outputs may be stale.
34. As a researcher, I want export to write a processed JPEG per paired segment, so that I have training images on disk.
35. As a researcher, I want export to write a matching `.txt` file per paired segment, so that each image has ground-truth text alongside it.
36. As a researcher, I want exported files named `<page_stem>_<segment_number>`, so that outputs are distinguishable and sortable.
37. As a researcher, I want unpaired segments skipped on export, so that partial progress is allowed.
38. As a researcher, I want warnings listing unpaired segments and unused text lines on export, so that I notice incomplete pages.
39. As a researcher, I want polygon segments rectified to a rectangle image on export, so that line crops are axis-aligned for OCR training.
40. As a researcher, I want rectangle segments rectified consistently on export, so that all outputs go through the same pipeline.
41. As a researcher, I want processing implemented as pluggable steps, so that binarization or height normalization can be added later without re-annotating.
42. As a researcher, I want to re-export after editing segments or pairings, so that training files can be refreshed.
43. As a researcher, I want exported line images and text files in dedicated output directories, so that my source pages and transcriptions stay untouched.

### Developer and quality

44. As a developer, I want OpenAPI-generated types from the FastAPI schema, so that the Next.js client stays aligned with the API.
45. As a developer, I want the processing pipeline testable without the UI, so that rectify logic can be verified in isolation.
46. As a developer, I want the filesystem data layout documented, so that I know where to place inputs and find outputs.
47. As a researcher, I want Greek Unicode text handled correctly in transcriptions and exports, so that polytonic Greek is not corrupted.

## Implementation Decisions

### Architecture

- **Standalone annote app** in `annote/frontend` (Next.js App Router) and `annote/backend` (FastAPI). Not coupled to the Kalamos platform API, Postgres, users, or jobs.
- **Filesystem persistence** under `annote/data/` with no database.
- **Build order**: UI and canvas first, then annotation API persistence, then processing and export.
- Canvas drawing logic may be adapted from Kalamos legacy `ImageCanvas` concepts, but the annote shell is greenfield Next.js.

### Deep modules (testable interfaces)

| Module | Responsibility | Interface sketch |
|--------|----------------|------------------|
| **Data layout** | Resolve paths for pages, transcriptions, annotations, exports; ensure directories exist | `resolve_page_paths(stem)`, `list_pages()` |
| **Page catalogue** | Discover page images and optional transcriptions | `list_pages() -> PageSummary[]`, `load_page_transcription(stem) -> str` |
| **Text line parser** | Split page transcription into ordered text lines | `split_text_lines(text) -> TextLine[]` |
| **Annotation store** | Load/save segment geometry and pairings per page | `load_annotation(stem) -> PageAnnotation`, `save_annotation(stem, data)` |
| **Segment model** | Segment types (polygon, rectangle), creation-order numbering | Typed segment DTOs with `id`, `number`, `kind`, `points`, `paired_text_line_index` |
| **Export state** | Compare annotation revision vs last export timestamp/hash | `is_export_dirty(stem) -> bool`, `mark_exported(stem)` |
| **Processing pipeline** | Run ordered steps on a segment crop | `process(image, segment, steps: list[str]) -> Image` |
| **Rectify step** | Polygon/rectangle to axis-aligned line image | `rectify(page_image, segment) -> ndarray` |
| **Export service** | Export paired segments, collect warnings | `export_page(stem) -> ExportResult` |
| **Page API** | HTTP layer over catalogue + image serving | REST routes for pages, images, transcriptions, annotations |
| **Canvas editor** | Draw/select/edit segments on page image | Client component: tools, zoom/pan, segment overlay |
| **Pairing panel** | Text line list + inline edit for selected segment | Client component bound to selected segment |

### On-disk layout

- `data/manuscripts/pages/` — source page images
- `data/transcriptions/pages/` — page transcription text files (matching stems)
- `data/annotations/pages/` — per-page JSON (segments + pairings + export metadata)
- `data/manuscripts/lines/` — exported processed JPEGs
- `data/transcriptions/lines/` — exported line `.txt` files

### API contracts (conceptual)

- `GET /pages` — list pages with stem, has_transcription, segment_count, export_dirty
- `GET /pages/{stem}` — page detail + annotation summary
- `GET /pages/{stem}/image` — serve page image bytes
- `GET /pages/{stem}/transcription` — page transcription text + parsed text lines
- `GET /pages/{stem}/annotation` — full annotation JSON
- `PUT /pages/{stem}/annotation` — replace annotation (autosave from editor)
- `POST /pages/{stem}/export` — run processing + write line outputs; return warnings

### Geometry

- Store segment coordinates in **natural image pixel space** (implementation detail; not a user-facing contract).
- Polygon: ordered list of `[x, y]` points (four or more).
- Rectangle: four corners as ordered points.

### Export rules

- Manual trigger only; UI shows dirty state when annotation is newer than last successful export.
- Export paired segments only; skip unpaired.
- Warn with counts/lists of unpaired segments and unused text lines.
- Segment number in filenames = creation order at time of segment creation (stable id per segment; number does not reshuffle on delete in v1 unless specified in implementation — prefer stable segment id in JSON, number assigned at creation and not reused).

### Session model

- Resumable: autosaved JSON reloads segments and pairings.
- Export is the checkpoint for training deliverables; pages remain editable afterward.

## Testing Decisions

**What makes a good test**: Assert externally visible behavior — files on disk, API response shapes, export output dimensions — not internal canvas or React state.

| Module | Test focus |
|--------|------------|
| **Text line parser** | Splits on line breaks; preserves Greek Unicode; handles trailing newline |
| **Data layout** | Resolves paths; lists pages; missing transcription handled |
| **Annotation store** | Round-trip save/load; empty page; paired and unpaired segments |
| **Export state** | Dirty after edit; clean after export; dirty again after re-edit |
| **Rectify step** | Known synthetic polygon produces expected output size/orientation |
| **Export service** | Paired-only files written; warnings for gaps; correct filenames |
| **Page API** | Integration tests with temp data directory |

**Prior art**: Kalamos backend tests use pytest with temp fixtures; follow the same patterns in `annote/backend/tests/`.

**UI**: Manual QA for canvas UX in early slices; optional Playwright later — not required for v1 AFK slices.

## Out of Scope

- Kalamos platform integration, Postgres, user authentication, JWT, jobs/workers
- Automatic text-to-segment alignment or OCR inference
- Manuscript grouping UI, project/dashboard hierarchy, multi-user collaboration
- Block-level layout (columns, regions above line level)
- Kraken or automatic segmentation inside annote
- Packaging as Electron/desktop app; cloud deployment
- Import/export to eScriptorium XML or TEI (deferred; Kalamos compatibility later)
- Processing steps beyond `rectify` in v1 (normalize_height, binarize planned as pluggable follow-ups)
- Upload UI for images (filesystem drop only in v1)
- Segment number reordering UI; reading-order enforcement
- Versioned exports or git-like history of annotations

## Further Notes

- Domain glossary: `annote/CONTEXT.md`
- Sample data: move existing JPEG into `data/manuscripts/pages/` and add matching `.txt` transcription.
- Rectangle draw gesture (exact click sequence) is TBD at implementation; slice 004 is marked HITL for canvas UX review.
- Primary user deliverables are processed `.jpg` + `.txt` per paired segment; annotation JSON is an intermediate persistence format.
