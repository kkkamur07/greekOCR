# Kanban

> Updated 2026-05-21 — full issue set (000–017); all links resolve

**WIP limits:** In progress ≤ **4** | Review ≤ **4** (AFK only)

**HITL (2):** [005](005-inference-catalog-bindings.md), [014](014-frontend-layout-editor.md) — you implement/review → **Done** (skip Review column).

**AFK (15):** agent → **Review** → you → **Done**.

## Ready (AFK)

- [ ] [003-documents-parts-media](003-documents-parts-media.md)

## Ready (HITL)

- [ ] [005-inference-catalog-bindings](005-inference-catalog-bindings.md)

## In progress

_(empty — 0/4)_

## Review

_(empty — 0/4)_

## Done

- [x] [000-platform-foundation](done/000-platform-foundation.md)
- [x] [001-user-auth-jwt](done/001-user-auth-jwt.md)
- [x] [002-projects-sharing](done/002-projects-sharing.md)
- [x] [004-job-runner](done/004-job-runner.md)
- [x] [012-nextjs-openapi-codegen](done/012-nextjs-openapi-codegen.md)

## Backlog

- [ ] [006-segment-job-kraken](006-segment-job-kraken.md)
- [ ] [007-segment-merge](007-segment-merge.md)
- [ ] [008-layout-edit-reset-api](008-layout-edit-reset-api.md)
- [ ] [009-transcribe-job-layers](009-transcribe-job-layers.md)
- [ ] [010-ground-truth-copy-edit-api](010-ground-truth-copy-edit-api.md)
- [ ] [011-access-public-published](011-access-public-published.md)
- [ ] [013-frontend-projects-documents](013-frontend-projects-documents.md)
- [ ] [014-frontend-layout-editor](014-frontend-layout-editor.md)
- [ ] [015-frontend-transcription-editor](015-frontend-transcription-editor.md)
- [ ] [016-frontend-jobs-panel](016-frontend-jobs-panel.md)
- [ ] [017-frontend-public-published-view](017-frontend-public-published-view.md)

## Parallelism cheat sheet

| When | Agent (AFK) | You (HITL) |
|------|-------------|------------|
| **Now** | **003** | **005** model paths & seeds |
| 003 + 005 done | **006** + **011** | — |
| 006 done | **007** + **009** | — |
| 007 + 009 done | **008** + **010** | — |
| 003 + 012 done | **013** | **014** (after 008) |
| 006 + 004 + 012 | **016** | — |
| 011 + 012 | **017** | — |
