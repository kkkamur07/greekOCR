# Issues

Lightweight issue tracker for grab-bag work. For day-to-day docs see
[`docs/README.md`](../docs/README.md). For repo hygiene (kanban drift, doc
links), see [`docs/repository-hygiene.md`](../docs/repository-hygiene.md).

---

## Backlog (2)

| ID | Title | Epic |
|----|-------|------|
| [029](backlog/029-ml-callback-replay-sweeper.md) | ML callback replay / sweeper | Platform jobs |
| [034](backlog/034-hf-dataset-staging-publish.md) | Hub dataset staging + publish | Hugging Face |

---

## Board

| File | Purpose |
|------|---------|
| [kanban.md](kanban.md) | Human-readable columns |
| [dag.md](dag.md) | Dependency graph (active issues only) |
| [board.json](board.json) | Machine-readable snapshot |

---

## Archive

Completed issues and PRDs live under [`done/`](done/README.md):

| Folder | Scope | Issues |
|--------|--------|--------|
| [done/platform/](done/platform/) | Nomicous app foundation + annote merge | 000–027 |
| [done/huggingface/](done/huggingface/) | Hub pull, publish, registry | 030–036 |
| [done/inference/](done/inference/) | OCR design, helper, packaging | 028, 037–041 |

---

## Adding issues

Create `issues/backlog/NNN-short-title.md` with frontmatter:

```yaml
---
id: "NNN"
title: "short-title"
type: AFK              # AFK | HITL
status: backlog        # backlog | ready | in_progress | review | done
blocked_by: []
parent_prd: "issues/done/.../prd-....md"
---
```

Then refresh [kanban.md](kanban.md) and [dag.md](dag.md).
