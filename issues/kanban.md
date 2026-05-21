# Kanban

> Regenerated 2026-05-21 (016, 017 reviewed → done)

**WIP limits:** In progress ≤ **4** | Review ≤ **5** (AFK only)

**HITL:** [005](005-inference-catalog-bindings.md) — **you** (lane B).

## Ready (AFK)

_(empty — 006 blocked on 005 HITL)_

## Ready (HITL)

- [ ] [005-inference-catalog-bindings](005-inference-catalog-bindings.md)

## In progress

_(empty — 0/4)_

## Review

_(empty — 0/5)_

## Done

- [x] [000-platform-foundation](done/000-platform-foundation.md)
- [x] [001-user-auth-jwt](done/001-user-auth-jwt.md)
- [x] [002-projects-sharing](done/002-projects-sharing.md)
- [x] [003-documents-parts-media](done/003-documents-parts-media.md)
- [x] [004-job-runner](done/004-job-runner.md)
- [x] [011-access-public-published](done/011-access-public-published.md)
- [x] [012-nextjs-openapi-codegen](done/012-nextjs-openapi-codegen.md)
- [x] [013-frontend-projects-documents](done/013-frontend-projects-documents.md)
- [x] [016-frontend-jobs-panel](done/016-frontend-jobs-panel.md)
- [x] [017-frontend-public-published-view](done/017-frontend-public-published-view.md)

## Backlog

- [ ] [006-segment-job-kraken](006-segment-job-kraken.md)
- [ ] [007-segment-merge](007-segment-merge.md)
- [ ] [008-layout-edit-reset-api](008-layout-edit-reset-api.md)
- [ ] [009-transcribe-job-layers](009-transcribe-job-layers.md)
- [ ] [010-ground-truth-copy-edit-api](010-ground-truth-copy-edit-api.md)
- [ ] [014-frontend-layout-editor](014-frontend-layout-editor.md)
- [ ] [015-frontend-transcription-editor](015-frontend-transcription-editor.md)

## Parallelism cheat sheet

| When | Agent (AFK) | You (HITL) |
|------|-------------|------------|
| **Now** | — | **005** (lane B) |
| 005 merged | **006** (lane C) | — |
| 006 + 008 path | — | **014** (layout UI) |
