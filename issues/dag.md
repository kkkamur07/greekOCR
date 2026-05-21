# Issue DAG

> Regenerated 2026-05-21

## Warnings

- **Frontmatter vs body drift:** several issues use `done/NNN-….md` in `blocked_by` but `issues/done/…` in `## Blocked by` — equivalent for humans; normalize if automating.

## Stats

| Metric | Count |
|--------|------:|
| Total issues | 18 |
| Done | 5 |
| Ready (AFK) | 0 |
| Ready (HITL) | 1 |
| Backlog | 10 |
| In progress | 0 |
| Review | 2 |

## Parallel lanes (ready now)

Up to **2** AFK lanes without approval (WIP in progress ≤ 4). Review **2/5**.

| Lane | Issues | Branch | Subagent |
|------|--------|--------|----------|
| **A** | [003](003-documents-parts-media.md) — **Review** | `feat/003-documents-parts-media` | `lane-documents-parts-media` |
| **B** | [005](005-inference-catalog-bindings.md) (HITL — **you**) | `work/005-inference-catalog` | `lane-inference-catalog` |
| **D** | [011](011-access-public-published.md) — **Review** | `feat/011-access-public` | `lane-access-public` |

After **003** + **005** merged → **006** (lane C) + **013** (lane F) in parallel.

| Lane | Issues (same branch) | Branch | Subagent |
|------|------------------------|--------|----------|
| **C** | 006 → 007 → 008 | `feat/006-segment-pipeline` | `lane-segment-pipeline` |
| **E** | 009 → 010 | `feat/009-transcribe-pipeline` | `lane-transcribe-pipeline` |
| **F** | 013 | `feat/013-frontend-projects` | `lane-frontend-projects` |
| **G** | 014 (HITL) | `work/014-frontend-layout` | `lane-frontend-layout` |
| **H** | 015 | `feat/015-frontend-transcription` | `lane-frontend-transcription` |
| **I** | 016 | `feat/016-frontend-jobs` | `lane-frontend-jobs` |
| **J** | 017 | `feat/017-frontend-public` | `lane-frontend-public` |

## Mermaid

```mermaid
flowchart TD
  I000["000 platform-foundation ✓"]
  I001["001 user-auth-jwt ✓"]
  I002["002 projects-sharing ✓"]
  I004["004 job-runner ✓"]
  I012["012 openapi-codegen ✓"]
  I005["005 inference-catalog HITL"]
  I003["003 documents-parts-media"]
  I006["006 segment-job-kraken"]
  I007["007 segment-merge"]
  I008["008 layout-edit-reset-api"]
  I009["009 transcribe-job-layers"]
  I010["010 ground-truth-copy-edit"]
  I011["011 access-public-published"]
  I013["013 frontend-projects-docs"]
  I014["014 frontend-layout-editor HITL"]
  I015["015 frontend-transcription-editor"]
  I016["016 frontend-jobs-panel"]
  I017["017 frontend-public-view"]

  I000 --> I001
  I000 --> I004
  I000 --> I005
  I001 --> I002
  I001 --> I012
  I002 --> I003
  I003 --> I006
  I004 --> I006
  I005 --> I006
  I006 --> I007
  I007 --> I008
  I007 --> I009
  I005 --> I009
  I003 --> I009
  I009 --> I010
  I003 --> I011
  I001 --> I011
  I003 --> I013
  I002 --> I013
  I012 --> I013
  I008 --> I014
  I012 --> I014
  I010 --> I015
  I012 --> I015
  I004 --> I016
  I012 --> I016
  I006 --> I016
  I011 --> I017
  I012 --> I017

  classDef hitl stroke-dasharray: 5 5
  class I005,I014 hitl
```
