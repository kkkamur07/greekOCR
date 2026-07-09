# Issue DAG

> Updated 2026-07-09 — active backlog only

## Stats

| Metric | Count |
|--------|------:|
| Backlog | 1 |
| Ready | 1 |
| In progress | 0 |
| Done (archived) | 37 |

## Parallel lanes (ready now)

- **034** hf-dataset-staging-publish

## Mermaid

```mermaid
flowchart TD
  I029["029 ml-callback-replay-sweeper"]
  I034["034 hf-dataset-staging-publish"]
  I033["033 hf-publish-model-from-staging ✓"]
  I028["028 ocr-prediction-execution-design ✓"]
  I033 --> I034
  I028 --> I029
```

Historical DAG for issues 000–041: [done/README.md](done/README.md).
