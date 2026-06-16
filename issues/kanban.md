# Kanban

> Regenerated 2026-06-16

**WIP limits:** In progress <= **4** | Review <= **5** | Parallel AFK lanes without approval <= **2**

**Override:** User explicitly requested a broad parallel implementation wave; the completed wave is now in Review.

## Ready (AFK)

_(empty — 018 is in review)_

## Ready (HITL)

_(empty — 005 is in review)_

## In progress

_(empty)_

## Review

- [ ] [005-inference-catalog-bindings](005-inference-catalog-bindings.md)
- [ ] [018-annote-production-root](018-annote-production-root.md)
- [ ] [014-frontend-layout-editor](014-frontend-layout-editor.md)
- [ ] [019-authenticated-platform-shell](019-authenticated-platform-shell.md)
- [ ] [020-document-line-transcription-model](020-document-line-transcription-model.md)
- [ ] [021-editor-page-line-geometry](021-editor-page-line-geometry.md)
- [ ] [006-segment-job-kraken](006-segment-job-kraken.md)
- [ ] [007-segment-merge](007-segment-merge.md)
- [ ] [008-layout-edit-reset-api](008-layout-edit-reset-api.md)
- [ ] [009-transcribe-job-layers](009-transcribe-job-layers.md)
- [ ] [010-ground-truth-copy-edit-api](010-ground-truth-copy-edit-api.md)
- [ ] [022-page-transcription-pairing-progress](022-page-transcription-pairing-progress.md)
- [ ] [023-page-review-status](023-page-review-status.md)
- [ ] [024-annotation-history-restore](024-annotation-history-restore.md)
- [ ] [025-export-approved-line-artifacts](025-export-approved-line-artifacts.md)

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

- [ ] [015-frontend-transcription-editor](015-frontend-transcription-editor.md)
- [ ] [026-transcription-pdf-artifact](026-transcription-pdf-artifact.md)
- [ ] [027-remove-root-app-duplicates](027-remove-root-app-duplicates.md)
- [ ] [028-ocr-prediction-execution-design](028-ocr-prediction-execution-design.md)

## Parallelism Cheat Sheet

| When | Agent (AFK) | You (HITL) |
|------|-------------|------------|
| Now | 022, 023, 024, and 025 in review | — |
| 018 done | 019 and 020 already launched | — |
| 019 + 020 done | 021 | — |
| 021 done | 022 | — |
| 022 done | 023, 024, 025, 026 can run in parallel within WIP limits | 028 design can start |
| 023 + 024 + 025 + 026 done | 027 cleanup | — |
