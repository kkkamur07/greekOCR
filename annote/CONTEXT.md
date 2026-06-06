# annote

A standalone local browser app for manually segmenting manuscript page images and pairing each segment with its transcription, to produce training-ready line crops. Runs as FastAPI + React on localhost with filesystem storage (no database).

## Language

**Manuscript name**:
The filename stem of a Page image (no extension). Used as the prefix in exported line filenames to distinguish outputs.
_Avoid_: manuscript_name (code alias), codex id

**Page list**:
The simple home view: every image in `data/manuscripts/pages/`, sorted by filename. Click to open the editor. No manuscript grouping in v1.
_Avoid_: Project browser, codex navigator

**Manuscript**:
A named manuscript or codex (e.g. Grec_1360_CONSTANTINUS_Harmenopulus). Groups many Pages. Not surfaced in v1 UI.
_Avoid_: Document, project

**Page**:
A single manuscript image file — one folio or fragment photograph.
_Avoid_: Part, image (when meaning a stored file)

**Segment number**:
The 1-based index of a Segment on a Page, assigned in creation order (first drawn = 1). Used in exported filenames (`<manuscript_name>_<segment_number>`). Reading order does not need to match.
_Avoid_: Line number (reading-order index of text)

**Segment**:
A user-drawn region on a Page that isolates one line of text.
_Avoid_: Region, box, crop (crop is the output, not the annotation)

**Page transcription**:
The full ground-truth text for a Page, stored as one document with line breaks marking where one written line ends and the next begins.
_Avoid_: Transcription (alone), full text

**Text line**:
One line of text extracted from a Page transcription (split on line breaks).
_Avoid_: Line (ambiguous with Segment), row

**Segment**:
A user-drawn region on a Page that isolates one written line of ink. Drawn as either a **Polygon segment** or a **Rectangle segment**.
_Avoid_: Region, box, crop (crop is the output, not the annotation)

**Polygon segment**:
A Segment defined by free-form corner points (four or more) tracing the ink boundary of one written line.
_Avoid_: Mask, boundary (fine for export; avoid in UI copy)

**Rectangle segment**:
A Segment defined by four corners forming a rectangle (may be rotated relative to the Page). Placed by click-drag and clicking corners — not a single axis-aligned drag-only box.
_Avoid_: Bounding box, axis-aligned box

**Processed line image**:
The exported image for one Segment after processing (includes polygon-to-rectangle rectification; further steps TBD). Written as `<manuscript_name>_<segment_number>.jpg`.
_Avoid_: Line crop, crop (alone), raw segment

**Processing**:
A pluggable sequence of steps applied to a Segment's ink region before export. v1 implements **rectify** (polygon-to-rectangle) only; further steps (e.g. normalize height, binarize) are added later without changing annotation data.
_Avoid_: Preprocessing, inference

**Processing step**:
One named transformation in the Processing pipeline (e.g. `rectify`, `normalize_height`, `binarize`).
_Avoid_: Filter, transform (generic)

**Line transcription file**:
A single-line text file holding the paired transcription for one Segment. Exported alongside its Processed line image: `<manuscript_name>_<segment_number>.txt`.
_Avoid_: Text line (when meaning the in-memory string during editing)

**Export**:
Produce a Processed line image and Line transcription file for each paired Segment on a Page. Unpaired segments are skipped. Export warns about unpaired segments and unused Text lines. Triggered manually by the user.
_Avoid_: Save, download

**Export state**:
Whether a Page's on-disk exports are up to date with its current annotations and pairings. The UI shows when re-export is needed ("dirty").
_Avoid_: Unsaved, modified

## On-disk layout (Option D)

- `data/manuscripts/pages/` — full **Page** images
- `data/transcriptions/pages/` — full **Page transcription** files (same stem as the Page image)
- `data/annotations/pages/` — segment geometry and pairings per Page (`<page_stem>.json`)
- `data/manuscripts/export/` — exported **Processed line image**s and **Line transcription file**s together: `<manuscript_name>_<segment_number>.jpg` and `<manuscript_name>_<segment_number>.txt`

## Relationships

- A **Manuscript** contains one or more **Pages**
- A **Page** has exactly one **Page transcription**
- A **Page transcription** contains one or more **Text lines** (in reading order)
- A **Page** contains one or more **Segments**
- Each **Segment** is manually paired with at most one **Text line** (unpaired segments are not exported)
- A **Text line** should pair to at most one **Segment**; unused text lines trigger an export warning
- Pairing always starts by selecting a **Segment** on the canvas, then assigning its **Text line**
- Annotation happens on the full **Page** image; **Processed line image**s and **Line transcription file**s are **Export** artefacts
- A paired Segment produces one **Processed line image** and one **Line transcription file**, named `<manuscript_name>_<segment_number>`

**Pairing**:
The manual act of linking a selected Segment to its Text line. UI is hybrid: pick from the page's text line list or edit the text inline.
_Avoid_: Alignment, matching (implies automation)

## Example dialogue

> **Dev:** "I drew a polygon around two lines by mistake — is that one Segment or two?"
> **Domain expert:** "Two. A Segment is always one written line on the page."
>
> **Dev:** "The page transcription has 40 text lines — do I need automatic matching?"
> **Domain expert:** "No. I select a Segment first, then pick the matching Text line from the list or fix the wording inline."
>
> **Dev:** "If I close the browser mid-page, is the work lost?"
> **Domain expert:** "No — resume from saved annotations. Export is when the training images are ready."

## Build phases (implementation order)

1. **UI** — draw polygon/rectangle segments on full Page; select segment
2. **Backend** — persist annotations to JSON; serve page images and transcriptions
3. **Processing + Export** — polygon-to-rectangle and further processing; write `.jpg` + `.txt` per segment

Processing logic is pluggable; v1 UI/backend must not assume it is finished.

## Implementation notes (not domain)

annote frontend is **Next.js App Router** (greenfield in `annote/frontend/`). Canvas drawing logic may still be adapted from Kalamos's legacy `ImageCanvas`, but the app shell is not Vite.

annote backend is a **standalone FastAPI app** in `annote/backend/` — no users, jobs, or Postgres. Filesystem-only persistence under `annote/data/`. Not mounted on the Kalamos platform API.

## Outputs (what matters to the user)

Primary deliverables per paired Segment:

- `<manuscript_name>_<segment_number>.jpg` — **Processed line image**
- `<manuscript_name>_<segment_number>.txt` — paired transcription text

Intermediate files (`annotations/pages/<stem>.json`, page transcription sources) exist for the tool to work but are not the user's focus. Internal geometry encoding is an implementation detail.

**Session model**:
Annotation is resumable across sessions (autosaved JSON reloads segments and pairings). **Export** is the checkpoint for processed `.jpg`/`.txt` deliverables; pages may still be reopened and edited afterward.

## Flagged ambiguities

- "segment" vs Kalamos "Line" — same intent; Kalamos compatibility is deferred.
- "line" alone is ambiguous — use **Segment** (image) or **Text line** (transcription).
- Exact rectangle draw gesture (how many clicks before corners are editable) — TBD at implementation.
- Hybrid pairing UI may ship after core segment draw/select; workflow order (select Segment first) is fixed.
- annote is standalone today; export to Kalamos is a future concern, not current scope.
