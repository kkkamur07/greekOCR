# Documentation

Index for project docs. Start with [local development](guides/local-development.md) if you are running the stack on your machine.

---

## Guides

Day-to-day workflows.

| Doc | Topic |
|-----|--------|
| [guides/local-development.md](guides/local-development.md) | Docker Compose, local API, frontend, Supabase profile |
| [guides/learnings.md](guides/learnings.md) | Supabase, serverless (Vercel), Calamari training, frequent errors |
| [guides/testing.md](guides/testing.md) | Pytest lanes (unit, integration, ML) and full-suite commands |

---

## Deployment

Hosted environments and infrastructure.

| Doc | Topic |
|-----|--------|
| [deployment/supabase.md](deployment/supabase.md) | Supabase Postgres + Storage — env vars, migrations, Docker override |
| [deployment/database-roles.md](deployment/database-roles.md) | Least-privilege API, worker, and migration database roles |
| [deployment/production.md](deployment/production.md) | Production layout: Vercel (landing, app, API) + inference workers |
| [deployment/vercel-platform-api.md](deployment/vercel-platform-api.md) | Vercel platform API settings, Python runtime, bundle trimming |

Related deploy configs (not under `docs/`):

- [`deploy/platform/`](../deploy/platform/) — Vercel platform API bundle
- [`deploy/inference/`](../deploy/inference/) — Inference API + worker (Railway/Fly)

---

## Quality and audits

Cross-cutting reviews, hygiene, and hardening.

| Doc | Topic |
|-----|--------|
| [repository-hygiene.md](repository-hygiene.md) | Repo hygiene — docs drift, gitignore, rituals, improvement roadmap |
| [repository-cleanup-plan.md](repository-cleanup-plan.md) | Dead code and cleanup audit — confidence levels, PR series |
| [codebase-audit.md](codebase-audit.md) | Full-stack audit — critical issues, phased fix order (when present) |

---

## Frontend

Frontend performance guidance and the completed Next.js migration record.

| Doc | Topic |
|-----|--------|
| [frontend/performance-optimization.md](frontend/performance-optimization.md) | Image cache, thumbnails, SWR, rendering — phased PR order |
| [frontend/nextjs-migration.md](frontend/nextjs-migration.md) | Next.js App Router migration record and follow-up checklist |
| [frontend/reliability-accessibility.md](frontend/reliability-accessibility.md) | Cursor pagination, background job SSE subscriptions |

---

## Architecture decisions

| Doc | Topic |
|-----|--------|
| [adr/0001-browser-auth-memory-cookie-csrf.md](adr/0001-browser-auth-memory-cookie-csrf.md) | In-memory access tokens + HttpOnly cookie session + CSRF |
| [adr/0002-job-status-sse-with-polling-fallback.md](adr/0002-job-status-sse-with-polling-fallback.md) | SSE job updates with serverless-safe polling fallback |
| [adr/0003-image-canvas-archival-boundary.md](adr/0003-image-canvas-archival-boundary.md) | Separate archival decision for the legacy canvas family |

---

## Inference

Models, registry, and the local Inference Helper.

| Doc | Topic |
|-----|--------|
| [inference/adding-inference-models.md](inference/adding-inference-models.md) | Checklist: registry, Hub weights, platform catalog, helper sync |

Service README: [`inference/README.md`](../inference/README.md)
Helper packaging: [`packaging/helper/README.md`](../packaging/helper/README.md)

---

## App READMEs

| Doc | Scope |
|-----|--------|
| [`nomicous/frontend/README.md`](../nomicous/frontend/README.md) | Next.js App Router app setup and routes |
| [`nomicous/README.md`](../nomicous/README.md) | Platform app root |
| [`landing/README.md`](../landing/README.md) | Marketing site for nomicous.com |

---

## Domain context

Bounded-context glossaries (live next to the code they describe):

| Doc | Scope |
|-----|--------|
| [`nomicous/CONTEXT.md`](../nomicous/CONTEXT.md) | Platform annotation domain |
| [`inference/CONTEXT.md`](../inference/CONTEXT.md) | Inference registry, Hub vocabulary |

---

## Backlog

Tracked work lives in [`issues/`](../issues/) (kanban, backlog, archived issues).
