---
name: lane-segment-pipeline
description: AFK parallel lane C from issues/dag.md — issues 006, 007, 008 on one branch. Use on feat/006-segment-pipeline. Vertical-slice TDD; move each issue to Review only when lane slice is complete; do not delete branch at review limit if 007/008 remain.
---

You own **lane C — segment pipeline** (`issues/006-segment-job-kraken.md` → `007-segment-merge.md` → `008-layout-edit-reset-api.md`).

## Rules

- **Single branch for the whole lane:** `feat/006-segment-pipeline` (006, 007, 008 share this branch until merged)
- **Blocked until:** 003 merged (or branch from `feat/003-documents-parts-media`), 004 done, **005 HITL done** (inference catalog)
- **TDD:** one failing test → minimal code → green; FastAPI `TestClient` + real Postgres (`kalamos`)
- **DDD:** `backend/inference/` for adapters/jobs; segment merge in `backend/document/application/` or dedicated module per issue
- `PYTHONPATH=.`; no editable `greekocr` package install
- At **Review WIP limit (5)** with 008 still open: **keep branch**, pause, resume on same branch after human merges earlier issues

## Order on branch

1. **006** — segment job enqueue, Kraken adapter stub, canonical segment DTO, job poll
2. **007** — segment merge (`manual_geometry` preserved, prune cascade)
3. **008** — layout CRUD + reset layout API

## Done (per issue)

- Issue tests green; OpenAPI export if routes added
- Frontmatter `status: review`, `branch: feat/006-segment-pipeline`
- Update `issues/kanban.md` Review column (≤ 5 AFK in review)

Do not implement 009+ or lane D/E unless explicitly reassigned.
