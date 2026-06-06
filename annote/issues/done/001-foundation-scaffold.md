---
id: "001"
title: "foundation-scaffold"
type: AFK
status: done
blocked_by: []
parent_prd: "issues/prd.md"
---

## Parent PRD

`issues/prd.md` — Setup, architecture, developer stories

## What to build

Scaffold annote end-to-end dev shell: standalone FastAPI app in `annote/backend/` with health route and CORS for local Next.js; Next.js App Router app in `annote/frontend/` with a placeholder home page; documented `data/` directory layout (pages, transcriptions, annotations, line exports); README with two-terminal dev start; OpenAPI export + TypeScript codegen workflow stubbed or documented.

No annotation features yet — prove both processes start and the API responds.

## Error handling

- [x] Missing `data/` subdirectories are created on startup or documented as a setup step with clear errors.

## Dev / test data

- [x] Document expected folder layout in `annote/README.md`.
- [x] Optional: relocate sample JPEG into `data/manuscripts/pages/` as fixture for next slices.

## Acceptance criteria

- [x] `uvicorn` starts annote FastAPI; `GET /health` returns OK
- [x] `npm run dev` starts Next.js; home page loads
- [x] CORS allows frontend to call backend on localhost
- [x] `data/` layout matches PRD (pages, transcriptions, annotations, line outputs)
- [x] README documents how to run both services

## Blocked by

None — can start immediately.

## User stories addressed

- 1
- 46
