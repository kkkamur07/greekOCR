# Nomicous documentation

**Picking the work back up?** Read
[`resume-inference-redesign.md`](resume-inference-redesign.md) first. It is the entry point
for the 2026-08-04/05 inference redesign: where the trunk is, what was built, what red is
supposed to look like, and which actions are the owner's rather than an agent's.

Start with the audience-appropriate guide:

| Doc                                                     | Use it when                                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [Root README](../README.md)                             | You want the product overview and a quick start                                          |
| [Use and host Nomicous](guides/using-and-hosting.md)    | You want Docker, local inference, Supabase, or deployment steps                          |
| [Models and datasets](inference/models-and-datasets.md) | You want the runtime catalog, HTR/segmentation models, or dataset workflow               |
| [Technical architecture](architecture.md)               | You want the system design, stack rationale, jobs, notifications, or security boundaries |

## Developer guides

| Doc                                              | Scope                                              |
| ------------------------------------------------ | -------------------------------------------------- |
| [Local development](guides/local-development.md) | Existing service-by-service development reference  |
| [Testing](guides/testing.md)                     | Python, frontend, integration, ML, and CI commands |
| [Learnings](guides/learnings.md)                 | Operational lessons and known platform pitfalls    |

## Deployment and operations

| Doc                                                      | Scope                                                     |
| -------------------------------------------------------- | --------------------------------------------------------- |
| [Production deployment](deployment/production.md)        | Vercel surfaces, Supabase, workers, DNS, and rollback     |
| [Supabase](deployment/supabase.md)                       | Hosted Postgres, private Storage, poolers, and migrations |
| [Database roles](deployment/database-roles.md)           | Least-privilege service roles                             |
| [Vercel platform API](deployment/vercel-platform-api.md) | Python bundle and serverless constraints                  |
| [Release evidence](deployment/release-evidence.md)       | Per-release verification record                           |

## Inference and publishing

| Doc                                                             | Scope                                                             |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
| [Adding inference models](inference/adding-inference-models.md) | Registry, weights, platform catalog, and deployment checklist     |
| [Inference service](../inference/README.md)                     | Install from PyPI, releasing, contracts, limits, and runtime reference |
| [Hugging Face publishing](../scripts/hf/README.md)              | Model and dataset staging, validation, and upload workflow        |

## Platform references

- [`nomicous/CONTEXT.md`](../nomicous/CONTEXT.md) — domain terminology and
  annotation semantics
- [`database-design.md`](database-design.md) — schema, ownership, pooling,
  job state, `NOTIFY`, SSE, and polling
- [`nomicous/backend/README.md`](../nomicous/backend/README.md) — backend
  bounded contexts and routes
- [`nomicous/frontend/README.md`](../nomicous/frontend/README.md) — editor
  development and generated API client

## Status, handoff, and reviews

Point-in-time documents. They carry file paths and line numbers that drift — and, in the
review's case, paths that have since been deleted — so read them alongside the code rather
than as current reference.

| Doc                                                                             | Scope                                                                        |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [Resume: inference redesign](resume-inference-redesign.md)                       | **Current.** Trunk state, what was built, known gaps, and the owner's actions |
| [Merge handoff: inference redesign](merge-handoff-inference-redesign.md)         | History. Executed 2026-08-05; kept for *why* each conflict was resolved as it was |
| [Merge audit, 2026-08-05](merge-audit-2026-08-05.md)                             | History. Findings that produced the `fix/remediation-*` branches              |
| [Test suite review, 2026-08-05](test-suite-review-2026-08-05.md)                 | Which tests would fail if the behaviour they name broke — CI blind spots, tests that cannot fail, and coverage gaps |
| [Codebase review, 2026-08-04](codebase-review-2026-08-04.md)                     | Stale. Cleanup backlog and architectural candidates as believed that day      |

## Security

Security-specific records live under [`security/`](security/). They include
dependency vulnerability exceptions and their remediation plans.

## Internal backlog

Deferred work lives in the repository-root [`todo.md`](../todo.md) — P0/P1/P2, one
backlog, no second copy. It is an internal backlog, not a product capability list.
