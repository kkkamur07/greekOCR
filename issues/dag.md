# Issue DAG

> Regenerated 2026-07-09

## Warnings

- None

## Stats

| Metric | Count |
|--------|------:|
| Total | 37 |
| Ready Afk | 0 |
| Ready Hitl | 1 |
| Blocked | 7 |
| In Progress | 0 |
| Review | 1 |
| Done | 28 |

## Parallel lanes (ready now)

- _(none)_

## Mermaid

```mermaid
flowchart TD
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
  I015["015 frontend-transcription-editor ✓"]
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
  I026["026 transcription-pdf-artifact ✓"]
  I027["027 remove-root-app-duplicates ✓"]
  I028["028 ocr-prediction-execution-design HITL"]
  I029["029 ml-callback-replay-sweeper"]
  I030["030 hf-uri-resolve-and-cache"]
  I031["031 hf-local-bundled-offline-path"]
  I032["032 hf-remote-transcribe-tracer"]
  I033["033 hf-publish-model-from-staging"]
  I034["034 hf-dataset-staging-publish"]
  I035["035 hf-collection-sync"]
  I036["036 hf-registry-id-migration"]
  I000 --> I001
  I000 --> I004
  I000 --> I005
  I001 --> I002
  I001 --> I011
  I001 --> I012
  I002 --> I003
  I002 --> I013
  I003 --> I006
  I003 --> I009
  I003 --> I011
  I003 --> I013
  I004 --> I006
  I004 --> I016
  I005 --> I006
  I005 --> I009
  I006 --> I007
  I006 --> I016
  I007 --> I008
  I007 --> I009
  I008 --> I014
  I009 --> I010
  I010 --> I015
  I011 --> I017
  I012 --> I013
  I012 --> I014
  I012 --> I015
  I012 --> I016
  I012 --> I017
  I018 --> I019
  I018 --> I020
  I019 --> I021
  I019 --> I027
  I020 --> I021
  I021 --> I022
  I022 --> I023
  I022 --> I024
  I022 --> I025
  I022 --> I026
  I022 --> I028
  I023 --> I027
  I024 --> I027
  I025 --> I027
  I026 --> I027
  I028 --> I029
  I030 --> I031
  I030 --> I032
  I031 --> I032
  I032 --> I033
  I032 --> I036
  I033 --> I034
  I033 --> I035
classDef hitl stroke-dasharray: 5 5
```
