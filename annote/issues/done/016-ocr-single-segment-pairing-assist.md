---
id: "016"
title: "ocr-single-segment-pairing-assist"
type: AFK
status: done
blocked_by: []
parent_prd: "issues/prd-calamari-pairing-assist.md"
---

## Parent

`issues/prd-calamari-pairing-assist.md` — **Pairing assist** via Calamari **OCR prediction** (single-segment slice).

## What to build

End-to-end **OCR prediction** for one **Segment**, wired into the pairing UI:

- Extend **Segment** schema with `model_transcription` and `model_transcription_at` (optional, backward-compatible defaults).
- **Calamari OCR service**: lazy-loaded `Predictor` from configured checkpoint; optional `annote[calamari]` extra; rectified crop via existing **rectify** step → grayscale numpy → `predict_raw`.
- **Settings**: `ANNOTE_CALAMARI_CHECKPOINT` defaulting to `model/checkpoints/best.ckpt` at greekOCR repo root.
- On annotation save (`PUT`), clear `model_transcription` fields when a segment's `points` changed vs previous saved annotation.
- `POST /pages/{stem}/segments/{segment_id}/ocr` — run prediction, persist fields, return updated `PageAnnotation`. Allowed on locked pages.
- **SegmentPairingBar**: "OCR" button, read-only **Model transcription** display, "Use suggestion" → fills inline textarea (`text_override`). "Use suggestion" disabled when page locked.
- Regenerate OpenAPI / frontend types.
- Tests: OCR service with mocked predictor; API integration test; pairing bar Vitest for Use suggestion + locked disabled.

## Acceptance criteria

- [x] `Segment` includes `model_transcription` and `model_transcription_at`; absent fields load as null on legacy JSON
- [x] Single-segment OCR endpoint returns updated annotation with populated model fields
- [x] OCR uses rectified crop (same pipeline as segment preview / export rectify)
- [x] Changing segment `points` via annotation save clears that segment's model fields
- [x] Missing Calamari install returns clear error with install hint (no server crash)
- [x] Missing checkpoint path returns clear error
- [x] Pairing bar shows suggestion after OCR; "Use suggestion" sets inline text; disabled when page locked
- [x] OCR on locked page succeeds; accepting suggestion remains blocked
- [x] Unit + API tests pass with mocked predictor (no Calamari required in CI)

## Blocked by

None — can start immediately.

## User stories addressed

- 1–11, 18–21, 22–27 (single-segment and shared backend foundations)
