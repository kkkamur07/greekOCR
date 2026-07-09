---
id: "028"
title: "ocr-prediction-execution-design"
type: HITL
status: done
blocked_by:
  - "022-page-transcription-pairing-progress.md"
parent_prd: "issues/done/prd-annote-merge.md"
---

## Parent

`issues/done/prd-annote-merge.md` — Deferred OCR prediction execution design.

## Resolution (accepted)

- **Cloud path:** **Product job** per user OCR/segment action → **remote inference** → single callback → merge into Postgres (`nomicous/CONTEXT.md`).
- **Local path:** **Inference helper** on researcher machine; browser probes `localhost:8001`, calls `/inference/v1/run`, persists via hosted API. See ADR [`docs/decisions/002-local-inference-helper.md`](../../../docs/decisions/002-local-inference-helper.md) and [`prd-local-inference-helper.md`](prd-local-inference-helper.md).
- **Model layers:** **Model transcription** separate from **Ground truth**; accept/edit workflow unchanged.
- **Progress UI:** Job banner / `trackJobAndWait` for cloud; local path shows download/run status in editor.
- **Follow-up AFK issues:** 038 (local transcribe tracer), 039 (local segment tracer).

## Acceptance criteria

- [x] Decide whether selected-line OCR prediction is synchronous, job-backed, or both → **both hosts**: local `/run` or cloud **Product job**; not sync on cloud API.
- [x] Decide whether Page and Document OCR prediction use the existing job infrastructure → **cloud yes**; local uses browser-orchestrated `/run` + persist API.
- [x] Decide how Model transcription layers are named, stored, and compared to Ground truth → existing document-level **Transcription** + **Line transcription** model.
- [x] Decide how frontend progress and errors are shown for OCR prediction → job queue UI + local helper status messages.
- [x] Decide which model runtime pieces may be packaged for the annote backend while keeping the root model workspace separate → `inference/` at repo root; helper packages slim subset.
- [x] Produce follow-up AFK implementation issues once the design is accepted → 038, 039.

## User stories covered

- 28
- 29
- 50
