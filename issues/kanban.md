# Kanban

> Regenerated 2026-05-21 (live)

**WIP limits:** In progress ≤ **4** | Review ≤ **4** (AFK only)

**HITL (2):** 005, 014 — you implement/configure; skip Review → **Done**.

**AFK (16):** agent implements → **Review** → you accept → **Done**.

## Ready (AFK)

- [ ] [001-user-auth-jwt](001-user-auth-jwt.md) — unblocks after 000 merged
- [ ] [004-job-runner](004-job-runner.md) — parallel lane B after 000 merged

## Ready (HITL)

- [ ] [005-inference-catalog-bindings](005-inference-catalog-bindings.md) — after 000 merged

## In progress

_(empty — 0/4)_

## Review

- [ ] [000-platform-foundation](done/000-platform-foundation.md) — **1/4** — branch `feat/000-platform-foundation`

## Done

_(move here after you merge PR #…)_

## Backlog

### HITL (2)

- [ ] [005-inference-catalog-bindings](005-inference-catalog-bindings.md)
- [ ] [014-frontend-layout-editor](014-frontend-layout-editor.md)

### AFK (15 remaining)

- [ ] [001-user-auth-jwt](001-user-auth-jwt.md)
- [ ] [002-projects-sharing](002-projects-sharing.md)
- [ ] [003-documents-parts-media](003-documents-parts-media.md)
- [ ] [004-job-runner](004-job-runner.md)
- [ ] [006-segment-job-kraken](006-segment-job-kraken.md)
- [ ] [007-segment-merge](007-segment-merge.md)
- [ ] [008-layout-edit-reset-api](008-layout-edit-reset-api.md)
- [ ] [009-transcribe-job-layers](009-transcribe-job-layers.md)
- [ ] [010-ground-truth-copy-edit-api](010-ground-truth-copy-edit-api.md)
- [ ] [011-access-public-published](011-access-public-published.md)
- [ ] [012-nextjs-openapi-codegen](012-nextjs-openapi-codegen.md)
- [ ] [013-frontend-projects-documents](013-frontend-projects-documents.md)
- [ ] [015-frontend-transcription-editor](015-frontend-transcription-editor.md)
- [ ] [016-frontend-jobs-panel](016-frontend-jobs-panel.md)
- [ ] [017-frontend-public-published-view](017-frontend-public-published-view.md)

## Parallelism cheat sheet

| When | Agent (AFK) | You (HITL) |
|------|-------------|------------|
| **Now (review)** | **000** awaiting your review | — |
| 000 merged | **001** + **004** (2 lanes) | **005** model paths |
| 006 done | **007** + **009** | — |
| 007+009 done | **008** + **010** | — |
