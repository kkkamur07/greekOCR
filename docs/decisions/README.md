# Architecture decisions

Recorded decisions for greekOCR / Nomicous. Each ADR captures **context, problem, alternatives, decision, and consequences**.

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-platform-job-status-push.md) | Push platform job status to the browser (Postgres NOTIFY + SSE) | Proposed |
| [002](002-local-inference-helper.md) | Local inference via Inference Helper (browser calls localhost, API persists) | Accepted |
| [003](003-supabase-hosted-postgres-and-storage.md) | Supabase as hosted Postgres + Storage (Alembic + MediaStore) | Accepted |
| [004](004-production-hosting-vercel.md) | Production hosting: Vercel (web + API) + persistent inference workers | Accepted |

When an ADR is implemented, update its status to **Accepted** and check off the implementation checklist in the doc.
