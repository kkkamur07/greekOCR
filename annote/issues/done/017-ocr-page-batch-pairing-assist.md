---
id: "017"
title: "ocr-page-batch-pairing-assist"
type: AFK
status: done
blocked_by:
  - "016-ocr-single-segment-pairing-assist.md"
parent_prd: "issues/prd-calamari-pairing-assist.md"
---

## Parent

`issues/prd-calamari-pairing-assist.md` — page-level **OCR prediction** with streaming progress.

## What to build

Batch **Pairing assist** for an entire **Page** — end-to-end through API and editor header:

- Page OCR event iterator (mirror export stream pattern): yield `progress` per segment (`current`, `total`, `segment_number`, `segment_id`), then `done` with summary counts.
- `POST /pages/{stem}/ocr/stream` — NDJSON stream; processes **all segments** on the page in order; updates `model_transcription` + `model_transcription_at` on each; persists annotation (incremental or final save — document chosen policy in tests).
- Zero segments: stream completes with `done` and zero count (no confusing error).
- **PageEditor** header: "OCR page" button with in-flight disabled state and progress label (reuse export progress UX where possible).
- Frontend stream client (mirror `exportPage` NDJSON reader).
- Tests: stream integration test with mocked predictor (event count, final annotation fields on all segments).

Depends on Calamari service, schema, and settings from slice 016.

## Acceptance criteria

- [x] Page OCR stream emits one progress event per segment, then done
- [x] After page OCR, every segment on the page has `model_transcription` set (mock returns deterministic text per call)
- [x] Page with zero segments completes without error
- [x] Header button shows progress during stream; annotation state refreshes on completion
- [x] Page OCR allowed on locked pages; pairing accept rules unchanged from slice 016
- [x] Stream error event on failure; client surfaces message via toast
- [x] Integration test for stream passes without real Calamari install

## Blocked by

- `issues/016-ocr-single-segment-pairing-assist.md`

## User stories addressed

- 12–17, 21 (page-level batch and toolbar discoverability)
