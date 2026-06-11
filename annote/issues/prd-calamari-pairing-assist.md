# annote — Calamari OCR Pairing Assist PRD

## Problem Statement

Researchers using annote to segment manuscript pages and **Pair** each **Segment** to **Text lines** from a **Page transcription** must type or pick every line by hand. They have a trained Calamari line OCR model (`best.ckpt`) that can read rectified line images, but annote offers no way to run that model on **Segments** during annotation.

Without **Pairing assist**, comparing model output to ground truth is slow: researchers leave annote, run external scripts, or guess at transcriptions while staring at ink. The PRD for core annote explicitly deferred OCR inference; researchers now need a **simple, on-demand** path to get a **Model transcription** suggestion per **Segment** (or for a whole **Page**) without automatic **Pairing** or overwriting ground truth.

Researchers who only want OCR on a subset of **Segments** can delete unwanted **Segments** before running page-level OCR — there is no separate segment-picker UI in v1.

## Solution

Add **Pairing assist** powered by Calamari **OCR prediction**:

- Run the model on the same **rectify**d line image used at **Export** time (polygon masked onto an axis-aligned rectangle).
- Trigger **on demand**: one **Segment** from the pairing bar, or all **Segments** on the **Page** from the editor header (with streaming progress like **Export**).
- Show the **Model transcription** as a read-only suggestion in the pairing UI; **Use suggestion** fills the inline pairing textarea (`text_override`) — same as typing. No automatic **Text line** assignment.
- Persist `model_transcription` and `model_transcription_at` on each **Segment** in annotation JSON; clear both when **Segment** geometry (`points`) changes.
- **OCR prediction** remains available on **Page lock**ed pages; accepting a suggestion is blocked when locked (pairing is frozen).
- Calamari is an **optional** backend extra (`annote[calamari]`), with a clear install hint when missing — same pattern as Kraken.
- Checkpoint path defaults to `model/checkpoints/best.ckpt` at the greekOCR repository root (sibling to `annote/`); override via `ANNOTE_CALAMARI_CHECKPOINT`.
- v1 returns **text only** — no confidence scores in the UI.

## User Stories

### OCR prediction and pairing assist

1. As a researcher, I want to run OCR on the selected segment while pairing, so that I get a suggested transcription without leaving annote.
2. As a researcher, I want OCR to use the same rectified line image as export, so that model input matches training data.
3. As a researcher, I want to see the model's suggested text separately from my paired text, so that I can compare before accepting.
4. As a researcher, I want a "Use suggestion" action that fills the inline pairing textarea, so that accepting a suggestion feels like typing it myself.
5. As a researcher, I want OCR suggestions to never change my pairing automatically, so that I stay in control of ambiguous lines.
6. As a researcher, I want to ignore an OCR suggestion and pick a text line from the page list instead, so that hybrid pairing still works.
7. As a researcher, I want to edit an accepted suggestion before saving, so that I can fix OCR errors inline.
8. As a researcher, I want the last OCR result for a segment saved in the annotation file, so that I can reopen the page without re-running the model.
9. As a researcher, I want saved OCR results cleared when I change segment geometry, so that stale suggestions do not mislead me after edits.
10. As a researcher, I want a clear error when Calamari is not installed, so that I know how to enable OCR assist.
11. As a researcher, I want a clear error when the checkpoint path is missing, so that I can fix configuration.

### Page-level OCR

12. As a researcher, I want to run OCR on all segments on the page in one action, so that I can batch-suggest transcriptions while reviewing a folio.
13. As a researcher, I want progress feedback during page OCR (e.g. segment 3 of 12), so that long pages do not feel stuck.
14. As a researcher, I want page OCR to run on every segment currently on the page, so that I can compare model output to existing pairings.
15. As a researcher, I want to delete unwanted segments before page OCR, so that I control which lines are processed without a separate picker UI.
16. As a researcher, I want page OCR to update saved model transcriptions on each segment, so that results persist across sessions.
17. As a researcher, I want page OCR to succeed on a page with zero segments with a clear outcome, so that empty pages do not error confusingly.

### Locked pages and workflow

18. As a researcher, I want to run OCR on a locked page, so that I can still preview model output against frozen annotation.
19. As a researcher, I want "Use suggestion" disabled on locked pages, so that lock still prevents pairing changes.
20. As a researcher, I want per-segment OCR available from the pairing bar when a segment is selected, so that assist sits where I already enter text.
21. As a researcher, I want page-level OCR in the editor toolbar near other page actions, so that it is discoverable alongside export and auto-segment.

### Configuration and dependencies

22. As a developer, I want the checkpoint path configurable by environment variable, so that different models can be tested without code changes.
23. As a developer, I want Calamari as an optional dependency, so that annotators who only segment are not forced to install TensorFlow.
24. As a developer, I want the OCR service lazy-loaded and reused across requests, so that repeated predictions do not reload the model every time.

### Testing and quality

25. As a developer, I want OCR prediction testable without the UI, so that rectified-crop → text behaviour can be verified in isolation.
26. As a developer, I want API integration tests with a mocked predictor, so that CI does not require Calamari or GPU.
27. As a developer, I want backward-compatible annotation JSON when `model_transcription` fields are absent, so that existing pages load unchanged.

## Implementation Decisions

### Domain and schema

- Reuse glossary terms from `annote/CONTEXT.md`: **OCR prediction**, **Model transcription**, **Pairing assist**, **rectify**, **Processing**.
- Extend **Segment** with optional fields:
  - `model_transcription: str | null` (default null)
  - `model_transcription_at: str | null` — ISO-8601 timestamp of last successful **OCR prediction** for this segment (default null)
- When annotation save detects `points` changed for a segment (compared to previous on-disk annotation), set `model_transcription` and `model_transcription_at` to null for that segment.
- **OCR prediction** does not modify `paired_text_line_index`, `text_override`, or **Export** dirty state by itself.

### OCR service (backend)

- New service module wrapping Calamari `Predictor.from_checkpoint` with lazy singleton (same pattern as Kraken model cache).
- Input: RGB numpy page image + segment dict → apply existing **rectify** step → convert to grayscale `uint8` numpy → `predictor.predict_raw([image])` → extract sentence string.
- Optional dependency: `calamari-ocr` via `annote[calamari]` extra; `ImportError` → `RuntimeError` with install hint naming `pip install -e '.[calamari]'` from `annote/backend`.
- Settings nested class `CalamariSettings` on application `Settings`:
  - `checkpoint: Path` — default resolves to `<greekOCR-repo>/model/checkpoints/best.ckpt`; env `ANNOTE_CALAMARI_CHECKPOINT`
  - `device` optional (CPU default if unset; agent may mirror Calamari env conventions)

### API contracts

- `POST /pages/{stem}/segments/{segment_id}/ocr`
  - Loads page image and annotation; finds segment by id.
  - Runs **OCR prediction**; writes `model_transcription` + `model_transcription_at` on that segment; saves annotation JSON.
  - Returns updated `PageAnnotation` (or segment-only DTO — prefer full annotation for simple client refresh).
  - 404 page/segment not found; 400/503 when Calamari missing or checkpoint invalid; allowed when page locked.
- `POST /pages/{stem}/ocr/stream`
  - NDJSON stream (same media type pattern as export stream).
  - Events: `progress` (`current`, `total`, `segment_number`, `segment_id`), `done` (`result`: counts), `error` (`detail`).
  - Iterates all segments on page in stable order (segment number / list order).
  - Persists each segment's model fields before yielding progress; final save once at end (or incremental saves — agent chooses; must survive partial completion policy documented in issue).
  - Allowed when page locked.
- Regenerate OpenAPI; update frontend generated types.

### Frontend

- **SegmentPairingBar** (pairing textarea area):
  - "OCR" button → calls single-segment endpoint; loading state on button.
  - Read-only suggestion block when `model_transcription` present (label: model suggestion / similar).
  - "Use suggestion" → `onTextOverride(model_transcription)`; disabled when page locked or no suggestion.
- **PageEditor** header:
  - "OCR page" button → calls stream endpoint with progress indicator (reuse export progress UX pattern).
  - Disabled when no segments or while OCR/export/segment in flight (agent-aligned with existing busy flags).
- After OCR completes, refresh annotation state from API response / refetch.
- Install hint surfaced via toast on 400/503 from API error body.

### Interaction with existing features

- **Export**, transcription PDF, preview export: unchanged; preview still shows rectify crop only.
- **Page lock**: OCR allowed; pairing textarea and "Use suggestion" follow existing locked disabled behaviour.
- **Annotation history**: OCR updates are annotation saves — timed/milestone snapshots follow existing `maybe_capture_on_save` policy if OCR endpoints call it (recommended: yes on persisted annotation update).
- **Kraken auto-segment**: no automatic OCR after segmentation in v1.

## Testing Decisions

**What makes a good test**: Assert externally visible behaviour — API status codes, annotation JSON fields, stream event sequence, UI disabled states when locked — not Calamari internal tensor shapes or React implementation details.

**Proposed test seams** (highest first):

1. **OCR service** — unit tests with monkeypatched `predict_raw` returning a fixed sentence; assert rectify path is exercised (can spy on `apply_step` / `process`).
2. **Stale model transcription** — unit test on annotation merge helper: changing `points` clears model fields; unchanged geometry preserves them.
3. **POST single-segment OCR API** — integration test via `TestClient` + temp `data_root` + mocked predictor; assert JSON fields and 404 for bad segment id.
4. **POST page OCR stream** — integration test: NDJSON progress lines count matches segment count; final annotation has all model fields set.
5. **SegmentPairingBar (UI)** — Vitest: "Use suggestion" calls `onTextOverride` with model text; disabled when `locked` prop true.
6. **PageEditor (UI)** — Vitest or manual QA: header button triggers stream client; progress label updates.

**Prior art**: `test_export_service.py` (streaming events), `test_kraken_segment.py` (optional ML + monkeypatch), `test_preview_service.py` (rectify crop), `SegmentPairingBar` / export progress patterns in `PageEditor.tsx`.

Please confirm these seams match your expectations before implementation slices start.

## Out of Scope

- Automatic **Pairing** or fuzzy match to **Text lines** from model output
- Auto-run OCR after Kraken **Auto-segment** or on segment select
- Confidence scores or character-level probabilities in UI
- Segment subset picker for page OCR (delete segments manually instead)
- Raw polygon crop or full-page PageXML-style OCR input
- Multiple checkpoints or model picker in UI
- Kalamos platform integration, background job queue, or multi-user OCR workers
- Writing model output directly to **Line transcription file** or **Export** artefacts without explicit **Pairing**
- GPU requirement in CI; training or fine-tuning Calamari inside annote
- Binarize / normalize_height processing steps before OCR (rectify only, same as export v1)

## Further Notes

- Parent glossary: `annote/CONTEXT.md` (**OCR prediction**, **Model transcription**, **Pairing assist**).
- Calamari API reference: `_support_repo/calamari` (`Predictor.from_checkpoint`, `predict_raw` on grayscale numpy).
- Trained checkpoint lives at `model/checkpoints/best.ckpt` (SavedModel directory layout).
- Vertical slices: issues `016` (single-segment assist), `017` (page batch + stream).
- Updates the main `issues/prd.md` out-of-scope line on OCR inference — **Pairing assist** is now in scope as a separate PRD, not retrofitted into the original monolithic PRD user-story numbering (new stories 104+ reserved in slice issues).
