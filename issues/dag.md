# Issue DAG

> Regenerated 2026-06-16

## Warnings

- Existing pre-merge issues `005` through `017` still reference `issues/prd.md`; new annote merge issues `018` through `028` reference `issues/prd-annote-merge.md`.
- Several older issues use `done/NNN-....md` in frontmatter and `issues/done/...` in body text; this is equivalent for humans but should be normalized before strict automation.

## Stats

| Metric | Count |
|--------|------:|
| Total issues | 29 |
| Done | 10 |
| Ready (AFK) | 0 |
| Ready (HITL) | 0 |
| Backlog | 7 |
| In progress | 0 |
| Review | 12 |

## Parallel lanes (ready now)

| Lane | Issues | Status |
|------|--------|--------|
| **A** | [018](018-annote-production-root.md) (AFK) | Review |
| **B** | [005](005-inference-catalog-bindings.md) (HITL — **you**) | Review |

## Parallel lanes (after blockers)

| Lane | Issues | Branch |
|------|--------|--------|
| **A1** | 019 → 021 → 022 → 023 | `feat/019-authenticated-platform-shell` |
| **A2** | 020 → 021 → 022 → 024 | `feat/020-document-line-transcription-model` |
| **A3** | 022 → 025 | `feat/025-export-approved-line-artifacts` |
| **A4** | 022 → 026 | `feat/026-transcription-pdf-artifact` |
| **A5** | 023 + 024 + 025 + 026 → 027 | `feat/027-remove-root-app-duplicates` |
| **H1** | 022 → 028 (HITL) | `work/028-ocr-prediction-execution-design` |
| **C** | 006 → 007 → 008 | `feat/006-segment-pipeline` |
| **E** | 009 → 010 | `feat/009-transcribe-pipeline` |
| **G** | 014 (HITL) | `work/014-frontend-layout` |
| **H** | 015 | `feat/015-frontend-transcription` |

## Mermaid

```mermaid
flowchart TD
  I000["000 platform-foundation ✓"]
  I001["001 user-auth-jwt ✓"]
  I002["002 projects-sharing ✓"]
  I003["003 documents-parts-media ✓"]
  I004["004 job-runner ✓"]
  I005["005 inference-catalog HITL"]
  I006["006 segment-job-kraken"]
  I007["007 segment-merge"]
  I008["008 layout-edit-reset-api"]
  I009["009 transcribe-job-layers"]
  I010["010 ground-truth-copy-edit"]
  I011["011 access-public-published ✓"]
  I012["012 openapi-codegen ✓"]
  I013["013 frontend-projects-docs ✓"]
  I014["014 frontend-layout-editor HITL"]
  I015["015 frontend-transcription-editor"]
  I016["016 frontend-jobs-panel ✓"]
  I017["017 frontend-public-view ✓"]
  I018["018 annote-production-root"]
  I019["019 authenticated-platform-shell"]
  I020["020 document-line-transcription-model"]
  I021["021 editor-page-line-geometry"]
  I022["022 page-transcription-pairing-progress"]
  I023["023 page-review-status"]
  I024["024 annotation-history-restore"]
  I025["025 export-approved-line-artifacts"]
  I026["026 transcription-pdf-artifact"]
  I027["027 remove-root-app-duplicates"]
  I028["028 ocr-prediction-execution-design HITL"]

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

  I018 --> I019
  I018 --> I020
  I019 --> I021
  I020 --> I021
  I021 --> I022
  I022 --> I023
  I022 --> I024
  I022 --> I025
  I022 --> I026
  I019 --> I027
  I023 --> I027
  I024 --> I027
  I025 --> I027
  I026 --> I027
  I022 --> I028

  classDef hitl stroke-dasharray: 5 5
  class I005,I014,I028 hitl
```
