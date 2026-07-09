---
id: "038"
title: "inference-helper-local-transcribe-tracer"
type: AFK
status: done
blocked_by: []
parent_prd: "issues/done/inference/prd-local-inference-helper.md"
triage: ready-for-agent
---

## Parent

`issues/done/inference/prd-local-inference-helper.md` — Local inference via **Inference helper** (browser-orchestrated).

## What to build

First end-to-end tracer for **local inference**: ship the slim **Inference helper** runtime and wire **Pairing assist** transcribe through it when the helper is healthy, the model has `host_eligibility: local`, and **Inference preference** is not forced to cloud.

The helper is a new slim FastAPI app (`python -m inference.helper`) binding `127.0.0.1:8001` with `GET /health`, `POST /inference/v1/run` (transcribe), and `GET /inference/v1/catalog` — no Postgres, no async jobs router, no service-secret auth on `/run` (v1). Cache dir defaults to `~/.nomicous/hf/cache/` via `HF_CACHE_ROOT`. CORS allows the hosted production origin and local dev.

Add `host_eligibility` to **Registry** entries; mark current transcribe models `local`. Catalog exposes eligibility to the frontend.

Add an authenticated hosted API endpoint to persist local transcribe results (line id, **registry model id**, text, confidence, character confidences) into the same **Model transcription** shape job callbacks produce.

Frontend: probe helper health (short timeout), read **Use cloud inference** preference (`localStorage` v1), and in pairing assist call localhost `/run` then persist via API when local-eligible; otherwise keep existing `enqueueTranscribePart` + `trackJobAndWait`.

## Acceptance criteria

- [x] `python -m inference.helper` serves `GET /health` and `POST /inference/v1/run` transcribe on `127.0.0.1:8001` without inference service secret.
- [x] Helper uses `HF_CACHE_ROOT` (default `~/.nomicous/hf/cache/`) and reuses existing weight resolution for `hf://`, `file://`, and `package://`.
- [x] `host_eligibility` is defined on **Registry** models; transcribe models are `local`; catalog endpoint returns eligibility.
- [x] Authenticated persist endpoint writes **Model transcription** rows equivalent to cloud transcribe job merge for a single line.
- [x] Pairing assist uses local path when helper is up, model is `local`, and cloud toggle is off; otherwise uses **Product job** + **remote inference**.
- [x] **Use cloud inference** toggle exists and forces cloud path on next pairing assist run.
- [x] Integration tests cover helper `/run` transcribe and persist API; registry/catalog unit tests cover `host_eligibility`.

## Blocked by

None — can start immediately.

## User stories covered

- 11, 12, 16, 17, 18, 19, 21, 23, 24, 25, 27, 32, 33, 34, 35, 36, 38, 41, 46, 47, 49
