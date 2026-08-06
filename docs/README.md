# Nomicous documentation

Start with whichever guide matches what you are here for:

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

| Doc                                                             | Scope                                                                  |
| --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [Adding inference models](inference/adding-inference-models.md) | Registry, weights, catalog, helper sync, and deployment checklist      |
| [Inference service](../inference/README.md)                     | Install from PyPI, releasing, contracts, limits, and runtime reference |
| [Hugging Face publishing](../scripts/hf/README.md)              | Model and dataset staging, validation, and upload workflow             |

## Platform references

- [`nomicous/CONTEXT.md`](../nomicous/CONTEXT.md) for domain terminology and
  annotation semantics
- [`database-design.md`](database-design.md) for the schema, ownership, pooling,
  job state, `NOTIFY`, SSE, and polling
- [`nomicous/backend/README.md`](../nomicous/backend/README.md) for backend
  bounded contexts and routes
- [`nomicous/frontend/README.md`](../nomicous/frontend/README.md) for editor
  development and the generated API client

## Security

Security-specific records live under [`security/`](security/). They cover dependency
vulnerability exceptions and their remediation plans.

## Internal backlog

Deferred work lives in [`todo.md`](todo.md). It is an internal backlog, not a product
capability list.
