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
A pluggable sequence of steps applied to a Segment's ink region before export. v1 implements **rectify** (polygon-to-rectangle) only; further steps (e.g. normalize height, binarize) are added later without changing annotation data. Distinct from **OCR prediction** (model transcription of a line image).
_Avoid_: Preprocessing (generic image ops)

**OCR prediction**:
Running a trained line OCR model (Calamari) on one Segment's line image to produce a **Model transcription**. Uses the same **rectify** step as **Export** (polygon masked onto an axis-aligned rectangle) so the model sees the same kind of line image it was trained on. On-demand assist only — does not change **Pairing** unless the researcher explicitly accepts the result.
_Avoid_: Inference (too generic), auto-transcription

**Model transcription**:
The text string returned by **OCR prediction** for one Segment. Shown as a pairing suggestion; not ground truth until accepted into **Pairing**.
_Avoid_: Prediction (ambiguous with ML jargon alone), OCR output

**Pairing assist**:
Using **OCR prediction** to suggest a **Model transcription** while the researcher pairs a selected Segment. Triggered on demand — either for one selected Segment or for all Segments on the Page. The researcher may accept the suggestion (into inline text or by matching a **Text line**), edit it, or ignore it. No automatic pairing.
_Avoid_: Auto-pair, auto-match

**Processing step**:
One named transformation in the Processing pipeline (e.g. `rectify`, `normalize_height`, `binarize`).
_Avoid_: Filter, transform (generic)

**Line transcription file**:
A single-line text file holding the paired transcription for one Segment. Exported alongside its Processed line image: `<manuscript_name>_<segment_number>.txt`.
_Avoid_: Text line (when meaning the in-memory string during editing)

**Export**:
Produce a Processed line image and Line transcription file for each paired Segment on a Page. Unpaired segments are skipped. Export warns about unpaired segments and unused Text lines. Triggered manually by the user.
_Avoid_: Save, download

**Transcription PDF**:
A single-page PDF for one **Page**: a blank page the same size as the **Page** image, with paired transcription text placed at each **Segment**'s position (using segment geometry coordinates). Plain text only — no facsimile, highlights, boxes, or borders. Text is drawn horizontal inside each segment's axis-aligned bounding box, auto-sized to fit, on a white page with dark text (not configurable in v1). No manuscript facsimile appears in the PDF. Only **paired** segments appear (unpaired segments are omitted). If none are paired, the PDF is still a blank page at the same dimensions. Used to **review** pairings while editing and as a **shareable artefact** for others who do not use annote. Two modes: **preview** (live, regenerated from current annotation); **share** (frozen PDF written at **Page lock**, served on download until the Page is unlocked and re-locked). Distinct from **Export** (per-segment `.jpg`/`.txt` training crops).
_Avoid_: Export PDF, report, facsimile PDF

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
- A **Kraken** segment stores a **Kraken ceiling** (original boundary) plus refined `points` shown in the editor; the ceiling constrains **Auto-refine**, not hand-edits
- Each **Segment** is manually paired with at most one **Text line** (unpaired segments are not exported)
- **Segments** on the same **Page** must not have **Segment overlap**; edge/corner contact is allowed
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
>
> **Dev:** "These two line polygons share a border — is that overlap?"
> **Domain expert:** "No. Touching edges are fine. Overlap means the interiors share ink area — two segments claiming the same pixels."
>
> **Dev:** "Kraken's box is too fat — can we tighten it automatically?"
> **Domain expert:** "Yes — shrink inward to the ink, but never past the Kraken boundary. I'll fix stubborn ones by hand afterward."

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

**Page lock**:
A deliberate state on a **Page** where its full annotation (all **Segments** and **Pairing**s) is frozen — no draw, edit, delete, or re-pair until explicitly **unlocked**. **Export** and transcription PDF may still be regenerated from the locked annotation. Distinct from **Export state** (whether on-disk exports match current annotations). Triggered manually at any time, or via a prompt when **pairing progress** reaches 100% (user may accept or dismiss; accepting locks the page).
_Avoid_: Finalize, archive, read-only mode (too vague)

**Annotation history**:
A time-ordered sequence of saved snapshots of a Page's annotation. Used to **restore** a prior state when a mishap occurs (bad edit, accidental overwrite). Restoring replaces the current annotation with the chosen snapshot; distinct from **unlock**, which simply allows editing the current state again.
_Avoid_: Backup (implies full-disk copy), version control (implies git)

**History snapshot**:
One persisted copy of a Page's annotation at a point in time. Written every **five minutes** during active editing on a Page, and when **pairing progress** crosses **50%** or **100%** (paired segments ÷ total segments). Retention: at most **five** rolling timed snapshots per Page, plus **protected milestone snapshots** (50%, 100%, lock, unlock) that are not evicted when the timed cap is reached. Restoring or unlocking does not delete history.
_Avoid_: Checkpoint (reserved for optional manual save if added later)

**Pairing progress**:
How far a Page's **Pairing** work has gone: paired segments vs total segments on that Page. A segment counts as paired when linked to a **Text line** or given inline text. Drives the progress UI, history milestones at 50% and 100%, and is the primary signal for "annotation work complete."
_Avoid_: Completion percentage (ambiguous with export state)

**Segment overlap**:
Two **Segments** on the same **Page** whose polygon interiors share area. Shared edges or corner contact alone is permitted — only interior area counts.
_Avoid_: Touching (too vague), collision (implies physics)

**Overlap repair**:
Removing **Segment overlap** by subtracting the shared interior from a **neighbour Segment**'s polygon. The **selected Segment** keeps its shape; the neighbour yields. Where overlap existed, the neighbour's new edge follows the selected segment's boundary. Triggered manually (e.g. a button while a segment is selected).
_Avoid_: Merge, split (different operations)

**Segment source**:
Whether a **Segment**'s geometry came from the annotator (**manual**) or **Kraken** auto-segmentation. Distinct from **Polygon segment** vs **Rectangle segment** (shape kind).
_Avoid_: Origin, provenance (fine in code; avoid in UI copy)

**Segment refinement**:
Improving a **Kraken** segment's polygon by snapping it to actual ink edges while shrinking inward. The original **Kraken** boundary is the maximum extent — the refined polygon may only shrink inside it, never grow beyond it. The annotator may still edit vertices afterward.
_Avoid_: Auto-segment (that's Kraken itself), rectify (that's export-time)

**Refinement margin**:
How far inward from the **Kraken** boundary **Segment refinement** may shrink (in image pixels). v1: fixed **4 px** (tunable in the 2–5 px range). Later: adaptive per line from segment height.
_Avoid_: Padding (implies crop only), overlap gap (different concern)

**Auto-refine**:
**Segment refinement** that runs automatically immediately after **Kraken** auto-segmentation, before segments are shown in the editor.
_Avoid_: Post-process (too generic)

**Kraken ceiling**:
The original **Kraken** polygon stored alongside the refined segment geometry. Hard maximum extent during **Auto-refine** only — hand-edits are not clamped to it.
_Avoid_: Original boundary, max extent (fine in code)

**Refinement fallback**:
When **Segment refinement** cannot find a stable ink edge for one segment, that segment keeps the unrefined **Kraken** polygon as its editable `points`. **Auto-refine** continues for the rest of the page.
_Avoid_: Skip (implies omitting the segment), fail (implies aborting the page)

**Ink edge signal**:
What **Segment refinement** snaps to inside the **Kraken ceiling**. v1: grayscale luminance edges (e.g. Canny) within the cropped region.
_Avoid_: Threshold (implies binarization only)

**Contour simplification**:
Reducing refined polygon vertex count after **Segment refinement** (e.g. Douglas–Peucker, ~2 px tolerance) so segments remain hand-editable.
_Avoid_: Decimation (too implementation-specific for glossary)

**Kraken ceiling overlay**:
Optional dashed outline of the **Kraken ceiling** shown for the selected segment. Toggleable in the editor; off by default.
_Avoid_: Ghost boundary, max extent preview

**Session model**:
Annotation is resumable across sessions (autosaved JSON reloads segments and pairings). **Export** is the checkpoint for processed `.jpg`/`.txt` deliverables. A **Page** may be **Page lock**ed when annotation work is complete; locked pages are not editable until unlocked.

## Flagged ambiguities

- **Overlap repair** (manual clip of neighbour segments) — discussed, deferred in favour of **Segment refinement** via active contours inside Kraken boundaries.
- "segment" vs Kalamos "Line" — same intent; Kalamos compatibility is deferred.
- "line" alone is ambiguous — use **Segment** (image) or **Text line** (transcription).
- Exact rectangle draw gesture (how many clicks before corners are editable) — TBD at implementation.
- Hybrid pairing UI may ship after core segment draw/select; workflow order (select Segment first) is fixed.
- annote is standalone today; export to Kalamos is a future concern, not current scope.
