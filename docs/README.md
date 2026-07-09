# Documentation

Index for project docs. Start with [local development](guides/local-development.md) if you are running the stack on your machine.

---

## Guides

Day-to-day workflows.

| Doc | Topic |
|-----|--------|
| [guides/local-development.md](guides/local-development.md) | Docker Compose, local API, frontend, Supabase profile |
| [guides/testing.md](guides/testing.md) | Pytest lanes (unit, integration, ML) and full-suite commands |

---

## Deployment

Hosted environments and infrastructure.

| Doc | Topic |
|-----|--------|
| [deployment/supabase.md](deployment/supabase.md) | Supabase Postgres + Storage — env vars, migrations, Docker override |
| [deployment/production.md](deployment/production.md) | Production layout: Vercel (landing, app, API) + inference workers |
| [deployment/docker-build-optimization.md](deployment/docker-build-optimization.md) | Image sizes, multi-stage builds, `.dockerignore` |

Related deploy configs (not under `docs/`):

- [`deploy/platform/`](../deploy/platform/) — Vercel platform API bundle
- [`deploy/inference/`](../deploy/inference/) — Inference API + worker (Railway/Fly)

---

## Inference

Models, registry, and the local Inference Helper.

| Doc | Topic |
|-----|--------|
| [inference/adding-inference-models.md](inference/adding-inference-models.md) | Checklist: registry, Hub weights, platform catalog, helper sync |
| [inference/local-inference-helper-launch-readiness.md](inference/local-inference-helper-launch-readiness.md) | Helper launch assessment, rollout, rollback |

Service README: [`inference/README.md`](../inference/README.md)

---

## Architecture

Deep dives and migration notes.

| Doc | Topic |
|-----|--------|
| [architecture/calamari-vendored-architecture.md](architecture/calamari-vendored-architecture.md) | Vendored Calamari HTR, Docker layout, troubleshooting |
| [architecture/hf-registry-id-migration.md](architecture/hf-registry-id-migration.md) | Hugging Face registry ID migration |

---

## Decisions (ADRs)

Recorded architecture decisions: [decisions/README.md](decisions/README.md)

---

## Domain context

Bounded-context glossaries (live next to the code they describe):

| Doc | Scope |
|-----|--------|
| [`nomicous/CONTEXT.md`](../nomicous/CONTEXT.md) | Platform annotation domain |
| [`inference/CONTEXT.md`](../inference/CONTEXT.md) | Inference registry, Hub vocabulary |

---

## Other

| Doc | Topic |
|-----|--------|
| [todo.md](todo.md) | Scratch-pad backlog (not the issue tracker) |
| [`issues/`](../issues/) | Backlog, kanban, and archived issues |
| [`landing/README.md`](../landing/README.md) | Marketing site for nomicous.com |
