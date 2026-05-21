# Issue DAG

> Generated 2026-05-21 from `issues/**/*.md` frontmatter

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
```

## Parallel lanes (when blockers clear)

| Lane | Issues | Notes |
|------|--------|-------|
| **Now** | 003 (AFK), 005 (HITL) | After 002/000 done |
| After 003 + 005 | 006, 011 | Segment + access policy |
| After 006 | 007, 009 | Merge + transcribe jobs |
| After 007 + 009 | 008, 010 | Layout API + ground truth API |
| After 003 + 012 | 013 | Projects/documents UI |
| After 006 + 004 + 012 | 016 | Jobs panel |
| After 011 + 012 | 017 | Public view |

## Warnings

- None — all kanban links resolve to committed issue files.
