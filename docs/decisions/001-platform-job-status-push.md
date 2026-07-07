# ADR-001: Push platform job status to the browser (Postgres NOTIFY + SSE)

## Status

**Proposed** — documents the problem observed in development (July 2026) and the recommended fix. The frontend still polls today; this ADR is the target design.

## Date

2026-07-07

## Context

Nomicous runs **product jobs** (segment, transcribe, …) that may delegate work to the **inference service**. Users enqueue a job from the page editor, then wait until the UI can reload layout or transcription data.

### What users see today

While a job runs, API logs fill with repeated requests like:

```text
GET /jobs/5ad9793f-9d0f-4ba8-a08e-b5509aa2a3d8 HTTP/1.1" 200 OK
SELECT users.id … FROM users WHERE users.id = $1
SELECT jobs.id … FROM jobs WHERE jobs.id = $1
ROLLBACK
```

These arrive roughly every **250 ms** for the duration of the job. Segmentation on a large page image (e.g. 5616×3744, ~145 lines) can take **30–60+ seconds** in Docker on Apple Silicon because `inference-worker` runs as `linux/amd64` (QEMU emulation) with TensorFlow/Kraken on CPU.

The job often **has finished in inference** while the browser is still polling — because “done” in the product sense only happens after the inference callback is processed and segment merge commits.

### Common confusion: “Shouldn’t there be notifications?”

**Yes — but only between backend services today.**

| Layer | Mechanism | Who it serves |
|-------|-----------|---------------|
| `inference_jobs` queue | Postgres `NOTIFY` on channel `inference_jobs` | Wakes `inference-worker` when a new inference job is inserted |
| Platform `jobs` table | HTTP polling from the browser | Frontend learns status via `GET /jobs/{id}` |
| Platform job worker | Polls Postgres for `pending` jobs | API process claims and submits to inference |

The inference stack already mirrors a good pattern: **write row → `pg_notify` → dedicated worker LISTENs → work runs → HTTP callback to platform**. The browser was never wired into that pattern; it polls the REST API instead.

---

## Problem statement

1. **Wasted work** — Each poll hits auth + DB even when nothing changed. A 45 s segmentation job at 250 ms intervals ≈ **180 redundant round-trips**.
2. **Noisy logs** — SQLAlchemy `BEGIN` / `SELECT` / `ROLLBACK` on every poll obscures real errors during development.
3. **Misleading mental model** — Postgres NOTIFY exists in the repo, so it feels like polling is a bug. It is not a bug; **push to the browser was never implemented**.
4. **Latency** — UI updates only as fast as the poll interval (250 ms best case) after the DB row actually becomes terminal.
5. **Scale** — Multiple users running jobs multiplies poll load linearly. Multiple API replicas do not share poll state (each client polls one replica).

---

## Current architecture (as built)

### Two job tables, one user-visible job

Users track a **product job** (`jobs` table in platform Postgres). Inference has its own **inference job** (`inference_jobs` table). The user sees one job id returned from `POST …/segment` or `POST …/transcribe`.

```text
Browser                nomicous-api              inference-api        inference-worker
   |                        |                         |                      |
   |-- POST segment ------->|                         |                      |
   |<-- { job_id } ---------|                         |                      |
   |                        |-- claim pending job       |                      |
   |                        |-- POST /inference/v1/jobs>|                      |
   |                        |   (status → waiting)      |-- INSERT + NOTIFY -->|
   |                        |                           |                      |-- Kraken / Calamari
   |-- GET /jobs/{id} ----->| (every 250 ms)            |                      |
   |                        |<-- POST job-complete -----|----------------------|
   |                        |   merge + status → done   |                      |
   |-- GET /jobs/{id} ----->|                           |                      |
   |<-- status: done -------|                           |                      |
```

### Product job status lifecycle

| Status | Meaning |
|--------|---------|
| `pending` | Enqueued; platform worker has not claimed yet |
| `running` | Platform worker claimed; about to submit to inference |
| `waiting` | Submitted to inference; waiting for callback |
| `done` | Callback processed; result merged (segments or transcriptions) |
| `failed` | Error at platform or inference layer |

**Important:** Inference can be finished while the product job is still `waiting`. Polling during `waiting` is expected with the current design.

### Where status changes are written

All paths use sync SQLAlchemy sessions and `commit()`:

| Location | Transitions |
|----------|-------------|
| `backend/jobs/infrastructure/job_repository.py` | `pending` → `running` (claim), → `waiting`, → `done`, → `failed` |
| `backend/jobs/application/job_callback_service.py` | `waiting` → `done` or `failed` (after inference callback + merge) |

None of these emit browser push events today.

### Frontend polling (two speeds)

| Use case | Code | Interval |
|----------|------|----------|
| Block until job finishes (segment / transcribe actions) | `pollJobUntilTerminal` in `nomicous/frontend/src/utils/jobPolling.ts` | **250 ms** (`JOB_WAIT_POLL_INTERVAL_MS`) |
| Jobs notice / background panel | `useJobPolling` in `nomicous/frontend/src/hooks/useJobPolling.ts` | **1500 ms** (`JOB_NOTICE_POLL_INTERVAL_MS`) |

`usePageEditorJobQueue.trackAndWait` uses `waitForJob` → `pollJobUntilTerminal` for editor actions.

### Backend polling (platform worker)

The API lifespan starts `worker_loop` (`backend/jobs/infrastructure/worker.py`), which polls for `pending` jobs with backoff (`JOB_POLL_INTERVAL_SECONDS`, default 0.25 s). This is separate from browser polling and is acceptable for queue draining.

### Inference NOTIFY (reference implementation)

On `create_job`, inference executes:

```python
session.execute(
    text("SELECT pg_notify(:channel, :payload)"),
    {"channel": "inference_jobs", "payload": str(job.id)},
)
```

`inference/jobs/worker.py` LISTENs on that channel via `JobNotificationListener` (`inference/infrastructure/db.py`) so the worker wakes immediately instead of polling the queue.

---

## Two layers: detection vs delivery

Push architecture is often described as one idea; it is really **two** problems:

| Layer | Question | Options |
|-------|----------|---------|
| **1. Detection** | How does the API learn that `jobs.status` changed? | Poll DB; inline publish after `commit()`; **Postgres NOTIFY** |
| **2. Delivery** | How does the **browser** receive the update? | HTTP poll (today); **SSE**; WebSocket |

- **Postgres NOTIFY** does not talk to the browser. It only signals API-side listeners that a row changed.
- **SSE** does not replace NOTIFY. SSE is the HTTP mechanism that keeps one connection open and streams events to the client.

A complete solution uses **NOTIFY for detection** and **SSE for delivery** (or WebSocket if bidirectional messages are needed later).

---

## Alternatives considered

### A. Keep HTTP polling (status quo)

- **Pros:** Already shipped; trivial with multiple API replicas; no long-lived connections.
- **Cons:** Log noise; redundant DB/auth load; 250 ms best-case UI delay; does not match inference-worker patterns.

**Verdict:** Acceptable for early v1; not the long-term design.

### B. SSE only, publish inline after `commit()` (no NOTIFY)

After `mark_job_done`, `mark_job_waiting`, etc., call an in-process `job_event_bus.publish(job_id)` that SSE handlers subscribe to.

- **Pros:** Simplest code in single-process dev (`uvicorn --workers 1`).
- **Cons:** Breaks with multiple Uvicorn workers or multiple API containers: the worker that handles the inference callback may not hold the browser’s SSE connection. Would need sticky sessions or a shared bus (Redis) anyway.

**Verdict:** Fine for a spike; not sufficient for production topology.

### C. WebSocket for all realtime UI

- **Pros:** Bidirectional; one connection for many event types later.
- **Cons:** More plumbing than needed for one-way job status; reconnect/auth story is heavier than SSE.

**Verdict:** Defer unless we need client→server messages on the same channel.

### D. Postgres NOTIFY → API LISTEN → SSE to browser (**recommended**)

- **Pros:** Same pattern as inference-worker; DB commit is the source of truth; every API instance can LISTEN and push only to **its local** SSE subscribers; no cross-instance Redis required for v1 multi-replica.
- **Cons:** Requires a background LISTEN task per API process; SSE connections are long-lived; need heartbeat/timeout handling.

**Verdict:** **Recommended.**

### E. NOTIFY → Redis pub/sub → SSE

- **Pros:** Explicit cross-replica fan-out.
- **Cons:** Extra infrastructure for a problem NOTIFY + per-process SSE subscribers already solve when each client is pinned to one API instance (normal load balancer behavior).

**Verdict:** Consider only if we need fan-out beyond “subscriber on same replica.”

---

## Decision

Adopt **Postgres NOTIFY for platform job status changes** and **SSE for browser delivery**.

### Recommended end-state

```text
┌─────────────┐     EventSource      ┌──────────────┐     LISTEN      ┌──────────┐
│   Browser   │◄──── SSE ────────────│  nomicous-api │◄─── NOTIFY ────│ Postgres │
│             │  /jobs/{id}/events   │  (per worker) │   platform_jobs│  jobs    │
└─────────────┘                      └───────┬──────┘                └────▲─────┘
                                             │                             │
                                             │ HTTP callback               │ UPDATE + NOTIFY
                                             │                             │
                                      ┌──────▼──────┐                ┌─────┴─────┐
                                      │ inference-  │── callback ────►│  (merge)  │
                                      │ worker      │                 └───────────┘
                                      └─────────────┘
```

1. **On every terminal or meaningful platform job status change**, after `commit()`, run `pg_notify('platform_jobs', payload)` where `payload` is JSON, e.g. `{"job_id": "…", "status": "waiting"}`.
2. **Each API process** runs a background `JobNotificationListener` (reuse the pattern from `inference/infrastructure/db.py`) on channel `platform_jobs`.
3. **On NOTIFY**, look up in-memory SSE subscribers for that `job_id` and write an SSE event.
4. **Browser** opens `GET /jobs/{job_id}/events` (authenticated) instead of polling `GET /jobs/{job_id}` in a tight loop.
5. **Keep `GET /jobs/{id}`** for one-shot reads and backwards compatibility; deprecate tight polling in the frontend.

### Suggested NOTIFY emission points

| Function | Notify? |
|----------|---------|
| `claim_next_pending_job` → `running` | Yes |
| `mark_job_waiting` | Yes |
| `mark_job_done` | Yes |
| `mark_job_failed` | Yes |
| `JobCallbackService` merge → `done` / `failed` | Yes (or centralize in repository helpers so callback path uses the same `update_job_status` function) |

Prefer **one repository function** `set_job_status(job_id, status, …)` that commits and NOTIFYs, so callback and worker cannot forget to emit.

### SSE endpoint sketch

- Route: `GET /jobs/{job_id}/events`
- Auth: same as `GET /jobs/{job_id}` (current user must own the job)
- Response: `text/event-stream`
- Events: full `JobResponse` JSON on each change; send current snapshot immediately on connect; close stream after `done` or `failed` (or keep open for future updates — closing is simpler)
- Heartbeat: comment line every 15–30 s so proxies do not kill idle connections

### Frontend migration

| Component | Change |
|-----------|--------|
| `pollJobUntilTerminal` | Replace with `waitForJobViaSse` (fallback to poll if `EventSource` fails or `Accept` negotiation says so) |
| `useJobPolling` | Subscribe per active job id, or one multiplexed stream later |
| `usePageEditorJobQueue.trackAndWait` | Use SSE waiter |

### Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `PLATFORM_JOB_NOTIFY_CHANNEL` | `platform_jobs` | Postgres NOTIFY channel name |
| `JOB_SSE_HEARTBEAT_SECONDS` | `30` | SSE keepalive interval |

Mirror naming of `INFERENCE_WORKER_NOTIFY_CHANNEL` in inference settings.

---

## Consequences

### Positive

- Eliminates ~O(duration / 250ms) poll requests per active job
- Cleaner API logs during ML work
- Aligns platform job UX with inference queue design already in the repo
- UI updates as soon as the row commits, without waiting for the next poll tick
- Clear separation: NOTIFY = backend detection, SSE = browser delivery

### Negative / risks

- **Long-lived connections** — one SSE per watching tab per job; load balancers must not buffer SSE (disable proxy buffering for this route)
- **Multi-replica** — NOTIFY wakes all API processes; only the replica with the SSE subscriber sends data (others no-op). Requires clients to stay connected to the same replica (standard HTTP) or we add Redis later
- **Implementation work** — repository refactor, lifespan LISTEN task, SSE route, frontend migration, integration tests
- **Timeouts** — Kraken on emulated amd64 can exceed 120 s; SSE wait timeout must match `JOB_WAIT` / worker timeout settings (today `pollJobUntilTerminal` defaults to 120 s)

### Testing

- Unit: mock event bus; NOTIFY payload parsing
- Integration: enqueue job → SSE client receives `running` → `waiting` → `done` in order
- Regression: `GET /jobs/{id}` still works for clients that have not migrated

---

## What this ADR does **not** change

- **Inference worker NOTIFY** — stays as-is on `inference_jobs`
- **Platform worker polling for `pending` jobs** — still needed to claim queue work unless we also NOTIFY on insert (optional follow-up)
- **Inference HTTP callback** — still the handoff from inference to platform; NOTIFY fires after platform DB update, not instead of callback
- **Segmentation slowness in Docker** — architectural; fix with native ARM images, GPU, or smaller test images — not solved by SSE alone

---

## References

| Artifact | Location |
|----------|----------|
| Product job glossary | `nomicous/CONTEXT.md` — **Product job** |
| PRD job execution (poll until terminal) | `issues/prd.md` — *Job execution* |
| Frontend poll intervals | `nomicous/frontend/src/utils/jobPolling.ts` |
| Inference NOTIFY on create | `inference/infrastructure/job_repository.py` |
| Inference LISTEN helper | `inference/infrastructure/db.py` — `JobNotificationListener` |
| Platform job repository | `nomicous/backend/jobs/infrastructure/job_repository.py` |
| Callback merge → done | `nomicous/backend/jobs/application/job_callback_service.py` |
| Jobs HTTP routes (poll today) | `nomicous/backend/jobs/api/jobs.py` |

---

## Implementation checklist (when scheduled)

- [ ] Add `set_job_status(...)` (or equivalent) with `pg_notify` after commit
- [ ] Add `PLATFORM_JOB_NOTIFY_CHANNEL` to platform settings
- [ ] Start LISTEN background task in `backend/core/app.py` lifespan
- [ ] Add in-memory `JobStatusBroadcaster` (job_id → set of SSE queues)
- [ ] Add `GET /jobs/{job_id}/events` SSE route
- [ ] Frontend: `waitForJobViaSse`, migrate `trackAndWait` and `useJobPolling`
- [ ] Integration test: segment job lifecycle over SSE
- [ ] Update `issues/prd.md` and `nomicous/backend/README.md` when shipped
