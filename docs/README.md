# Documentation

Index for project documentation. **New contributors:** start with
[local development](guides/local-development.md), then skim
[database design](database-design.md) for how the platform fits together.

```text
docs/
  README.md                 ← you are here
  database-design.md        platform schema, jobs, NOTIFY/SSE
  guides/                   day-to-day workflows
  deployment/               hosted environments and release ops
  inference/                models, registry, helper
  security/                 VEX and security notes
  todo.md                   deferred work
```

---

## Getting started

| Doc | When to read |
|-----|----------------|
| [guides/local-development.md](guides/local-development.md) | Docker Compose, local API, frontend, Supabase profile |
| [guides/testing.md](guides/testing.md) | Pytest lanes (unit, integration, ML) and CI commands |
| [guides/learnings.md](guides/learnings.md) | Supabase, Vercel, Calamari training, frequent errors |

More: [guides/README.md](guides/README.md)

---

## Platform

| Doc | When to read |
|-----|----------------|
| [database-design.md](database-design.md) | Tables, jobs, NOTIFY/SSE, pooling, transaction rules |
| [`nomicous/CONTEXT.md`](../nomicous/CONTEXT.md) | Annotation domain glossary (lives next to code) |
| [`nomicous/README.md`](../nomicous/README.md) | Platform app root - backend, frontend, migrations |

App setup and routes: [`nomicous/frontend/README.md`](../nomicous/frontend/README.md)

---

## Deployment

Hosted environments, database roles, and release operations.

| Doc | When to read |
|-----|----------------|
| [deployment/production.md](deployment/production.md) | Going live - Vercel surfaces, workers, rollback |
| [deployment/supabase.md](deployment/supabase.md) | Hosted Postgres + Storage, migrations, Docker override |
| [deployment/database-roles.md](deployment/database-roles.md) | Least-privilege API, worker, migrator roles |
| [deployment/vercel-platform-api.md](deployment/vercel-platform-api.md) | Vercel Python API build, bundle trimming |
| [deployment/release-evidence.md](deployment/release-evidence.md) | Per-deploy evidence template (baselines, sign-off) |

Deploy configs (not under `docs/`):

- [`deploy/platform/`](../deploy/platform/) - Vercel platform API bundle
- [`deploy/inference/`](../deploy/inference/) - Inference API + worker (Railway/Fly)

More: [deployment/README.md](deployment/README.md)

---

## Inference

| Doc | When to read |
|-----|----------------|
| [inference/adding-inference-models.md](inference/adding-inference-models.md) | Registry, Hub weights, platform catalog, helper sync |

Service README: [`inference/README.md`](../inference/README.md)  
Domain glossary: [`inference/CONTEXT.md`](../inference/CONTEXT.md)  
Helper packaging: [`packaging/helper/README.md`](../packaging/helper/README.md)  
Hub scripts: [`scripts/hf/README.md`](../scripts/hf/README.md)

More: [inference/README.md](inference/README.md)

---

## Other READMEs

| Doc | Scope |
|-----|--------|
| [`landing/README.md`](../landing/README.md) | Marketing site (nomicous.com) |
| [`deploy/platform/README.md`](../deploy/platform/README.md) | Vercel API bundle |
| [`deploy/inference/README.md`](../deploy/inference/README.md) | Cloud inference deploy |

---

## Backlog

Tracked work: GitHub Issues on the repository. Deferred items also live in
[todo.md](todo.md).
