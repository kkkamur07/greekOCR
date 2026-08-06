# 0003. One job queue: the cloud worker claims like any paired device

- Status: Accepted
- Date: 2026-08-04
- Builds on: [0002](./0002-inference-cli-replaces-loopback-helper.md), which made a laptop
  claim jobs directly from the platform.

## Context

A cloud job currently crosses two databases and two workers:

```
browser -> jobs row (platform PG) -> platform worker claims
        -> HTTP POST /inference/v1/jobs
        -> inference_jobs row (inference PG) -> inference worker claims
        -> runs -> callback -> jobs row updated -> SSE
```

Seven steps. The path ADR 0002 gives a laptop takes four, across one database: enqueue, claim,
run, callback.

That is the whole problem. We were about to ship a simpler, better-instrumented pipeline for
researchers' laptops than the one our own servers use, and then maintain both. `inference_jobs`
is a mailbox between two processes we control, and its existence forces a second claim loop, a
second admission-control mechanism, a second set of advisory locks, and a second Postgres
dependency.

## Decision

Delete the second queue. The cloud worker claims over HTTP from the same endpoint a laptop
does, authenticated with a service credential instead of a device token.

There is now one inference agent implementation. Local and cloud are the same program with
different credentials and different uptime, rather than two code paths kept in parity by
discipline.

This extends upward the property that makes `run_model()` worth protecting: local and cloud
produce identical output because they are literally the same code. That now holds for the
whole job lifecycle, not just the model call.

What this removes:

|                                                                  | lines |
| ---------------------------------------------------------------- | ----- |
| `inference/infrastructure/` (db, ORM, job repository, settings)  | 399   |
| `inference/api/jobs.py`                                          | 42    |
| `inference/jobs/worker.py` (queue half)                          | 161   |
| `ml_client.py` (the HTTP hop)                                    | 76    |

It also removes the `inference_jobs` table, the inference service's Postgres dependency
(`psycopg2` and `sqlalchemy` leave the `inference` dependency group), its queue-admission
advisory locks, and the `inference-api` container. The registry endpoint the agent syncs from
is served by the platform on port 8000, not by that service.

## Rationale

The rule this follows: the platform already runs a job queue, a claim loop, a validated
callback contract, a stale sweep, and an SSE status channel, and runs them well. Anything the
agent needs that one of those already does is *used*, not rebuilt. That is why this whole layer
costs one new endpoint. Completion and failure are `JobCallbackRequest`, abandonment is the
existing sweep, and status delivery is the existing SSE path. The second queue existed because
the inference service reimplemented a queue the platform already had.

Cloud inference was already off when this was decided, so this is a deletion rather than a
migration: no live path to preserve, no cutover, no compatibility window. It is therefore the
*first* step of the work rather than the last, because it shrinks the surface every later step
has to touch.

On claiming over HTTP rather than directly from platform Postgres: a direct
`claim_next_pending_job` against the platform database would be faster, but it couples the
inference worker to the platform schema and requires database access from wherever workers
run. The README's reason for splitting the services, that *"workers can later run on different
resources (e.g. GPU nodes) without changing the HTTP contract"*, survives an HTTP claim and
does not survive a database claim. The separate *worker* was always defensible. The separate
*queue* was not.

## Consequences

- The claim endpoint moves onto the hot path for all inference, not just laptops. ADR 0001
  already constrains it: it must not take `Depends(get_db)`, because a 25s long-poll pins a
  pooled connection and exhausts the pool at roughly fifteen devices. With cloud workers
  polling too, that ceiling binds sooner. Cloud workers should use a short poll rather than a
  long one, since they are never idle for long and do not need the latency.
- A second cost centre becomes one. Queue admission, rate limiting, and stale sweeping stop
  having two implementations with two sets of defaults.
- The inference service is no longer independently deployable as a thing that accepts and
  queues work on its own. Given ADR 0002 defers standalone inference, we are not currently
  using that property. If it is ever needed, it returns as a standalone CLI mode reading local
  files, not as a second queue.
- The cloud worker inherits the agent's lease semantics (`DEVICE_LEASE_SECONDS`, 600s per ADR
  0002). A server that does not sleep will never trip it, but it is now the one timeout rather
  than one of two.
