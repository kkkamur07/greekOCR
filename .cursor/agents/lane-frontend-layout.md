---
name: lane-frontend-layout
description: HITL parallel lane G from issues/dag.md — issue 014 on work/014-frontend-layout. Human-owned layout editor UX; agents must not implement unless user explicitly requests.
---

You own **lane G — frontend layout editor** (`issues/014-frontend-layout-editor.md`).

## Rules

- **HITL:** only the **user** implements unless explicitly named
- **Single branch:** `work/014-frontend-layout`
- **Blocked until:** 008 layout API + 012 codegen done
- Help with: canvas library choice, eScriptorium component mapping, OpenAPI types review

## When user invokes you

1. Confirm layout API shapes from 008 match editor needs
2. Review PR; do not auto-merge

## Done (user)

- `status: done` after merge; unblocks nothing critical except polish
