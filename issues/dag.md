# Issue DAG

> Regenerated 2026-06-16

## Warnings

- Existing pre-merge issues `005` through `017` still reference `issues/prd.md`; new annote merge issues `018` through `028` reference `issues/prd-annote-merge.md`.

## Stats

| Metric | Count |
|--------|------:|
| Total issues | 29 |
| Done | 27 |
| Ready (AFK) | 0 |
| Ready (HITL) | 1 |
| Backlog | 0 |
| In progress | 0 |
| Review | 1 |

## Parallel lanes (ready now)

| Lane | Issues | Status |
|------|--------|--------|
| _none_ | _No AFK lanes ready_ | _At WIP limit or awaiting review_ |

## Mermaid

```mermaid
flowchart TD
  I015["015 frontend-transcription-editor"]
  I026["026 transcription-pdf-artifact ✓"]
  I027["027 remove-root-app-duplicates review"]
  I028["028 ocr-prediction-execution-design HITL"]
  I000["000 platform-foundation ✓"]
  I001["001 user-auth-jwt ✓"]
  I002["002 projects-sharing ✓"]
  I003["003 documents-parts-media ✓"]
  I004["004 job-runner ✓"]
  I005["005 inference-catalog-bindings ✓"]
  I006["006 segment-job-kraken ✓"]
  I007["007 segment-merge ✓"]
  I008["008 layout-edit-reset-api ✓"]
  I009["009 transcribe-job-layers ✓"]
  I010["010 ground-truth-copy-edit-api ✓"]
  I011["011 access-public-published ✓"]
  I012["012 nextjs-openapi-codegen ✓"]
  I013["013 frontend-projects-documents ✓"]
  I014["014 frontend-layout-editor ✓"]
  I016["016 frontend-jobs-panel ✓"]
  I017["017 frontend-public-published-view ✓"]
  I018["018 annote-production-root ✓"]
  I019["019 authenticated-platform-shell ✓"]
  I020["020 document-line-transcription-model ✓"]
  I021["021 editor-page-line-geometry ✓"]
  I022["022 page-transcription-pairing-progress ✓"]
  I023["023 page-review-status ✓"]
  I024["024 annotation-history-restore ✓"]
  I025["025 export-approved-line-artifacts ✓"]
  I010 --> I015
  I012 --> I015
  I022 --> I026
  I019 --> I027
  I023 --> I027
  I024 --> I027
  I025 --> I027
  I026 --> I027
  I022 --> I028
  I000 --> I001
  I001 --> I002
  I002 --> I003
  I000 --> I004
  I000 --> I005
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
  I001 --> I012
  I003 --> I013
  I012 --> I013
  I002 --> I013
  I008 --> I014
  I012 --> I014
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
  classDef hitl stroke-dasharray: 5 5
  class I028,I005,I014 hitl
```
