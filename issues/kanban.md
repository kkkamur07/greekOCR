# Kanban

> Updated 2026-05-21 — review column cleared (000, 001, 002, 004, 012 merged to `main`)

**WIP limits:** In progress ≤ **4** | Review ≤ **4** (AFK only)

**HITL:** 005, 014 — no issue files yet; scope in `prd.md` until created.

**AFK:** agent implements → **Review** → you accept → **Done**.

## Ready (AFK)

- [ ] [003-documents-parts-media](003-documents-parts-media.md) — next backend slice (002 merged)

## Ready (HITL)

- [ ] **005** inference catalog & bindings — *issue file not created yet* (`prd.md`)

## In progress

_(empty — 0/4)_

## Review

_(empty — 0/4)_

## Done

- [x] [000-platform-foundation](done/000-platform-foundation.md) — merged `main`
- [x] [001-user-auth-jwt](done/001-user-auth-jwt.md) — merged `main`
- [x] [002-projects-sharing](done/002-projects-sharing.md) — merged `main`
- [x] [004-job-runner](done/004-job-runner.md) — merged `main`
- [x] [012-nextjs-openapi-codegen](done/012-nextjs-openapi-codegen.md) — merged `main`

## Backlog

### Has issue file (blocked)

- [ ] [011-access-public-published](011-access-public-published.md) — after 003

### Planned (no `issues/NNN-*.md` yet)

006 segment-job-kraken · 007 segment-merge · 008 layout-edit-reset-api · 009 transcribe-job-layers · 010 ground-truth-copy-edit-api · 013 frontend-projects-documents · 014 frontend-layout-editor (HITL) · 015 frontend-transcription-editor · 016 frontend-jobs-panel · 017 frontend-public-published-view

## Parallelism cheat sheet

| When | Agent (AFK) | You (HITL) |
|------|-------------|------------|
| **Now** | **003** documents/parts/media | **005** model catalog (when issue exists) |
| 003 done | **006** + **011** (if 005 done) | — |
| 006 done | **007** + **009** | — |
| 007+009 done | **008** + **010** | — |
