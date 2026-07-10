# Deployment documentation

Hosted environments, database operations, and release evidence.

| Doc | When to read |
|-----|----------------|
| [production.md](production.md) | **Primary runbook** — Vercel (landing, app, API), workers, rollback |
| [supabase.md](supabase.md) | Hosted Postgres + Storage, env vars, migrations |
| [database-roles.md](database-roles.md) | Least-privilege API, worker, and migrator roles |
| [vercel-platform-api.md](vercel-platform-api.md) | Vercel Python runtime, bundle size, dependency trimming |
| [nomicous-backend-hardening.md](nomicous-backend-hardening.md) | Backend security and boundary hardening |
| [release-evidence.md](release-evidence.md) | Per-deploy checklist — baselines, smoke tests, sign-off |

**Schema and jobs:** [database-design.md](../database-design.md)  
**Launch audit:** [codebase-audit.md](../codebase-audit.md)

Deploy configs outside `docs/`:

- [`deploy/platform/`](../../deploy/platform/) — Vercel API bundle
- [`deploy/inference/`](../../deploy/inference/) — Inference API + worker

Back to [docs index](../README.md).
