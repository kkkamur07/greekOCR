# 0005. The claim endpoint, and who owns a hosted worker's device row

- **Status:** Accepted
- **Date:** 2026-08-04
- **Builds on:** [0001](./0001-outbound-helper-device-pairing.md), which built the
  device layer and left the claim protocol to a later record;
  [0003](./0003-single-job-queue-cloud-worker-claims-like-a-device.md), which
  decided that the cloud worker claims from the same endpoint a laptop does.

## Context

ADR 0003 costs the platform exactly one new endpoint. This record is that
endpoint, plus the four decisions it forces that 0001 and 0003 do not already
imply.

`POST /device/v1/jobs/claim` hands one **inference agent** one page of work.
Completion and failure are the existing `JobCallbackRequest`. Abandonment is the
existing stale sweep. There is no heartbeat endpoint and no release endpoint.

## Decisions

### 1. The credential fixes the execution target; the caller cannot ask

An `X-Nomicous-Device-Token` claims `local` work belonging to the one account on
`helper_devices.user_id`. An `X-Nomicous-Service-Token` claims `cloud` work for
any account. Nothing in the request body can widen either.

A device token is deliberately **not** allowed to claim `cloud` work even when
its own `inference_host` says `cloud`. The two are different questions:
`inference_host` says which host's **capacity** a row reports, and a device
credential's entire authorization scope is a single `user_id` foreign key.
`cloud` work has no such owner, so honouring a device token for it would hand one
researcher's laptop every account's pages — and the row that would do it can be
created by any code path that writes a device.

This is the same asymmetry `hosts_with_recent_devices` already encodes: a laptop
answers for its owner, a hosted worker answers for everyone.

### 2. A claimed page becomes `waiting`, not `running`

`waiting` already means "an inference host holds this job and we are waiting for
its callback". With the second queue gone, claiming *is* the dispatch, so the
claim writes what the old dispatch wrote — status, `claimed_by`, and the
`inference_job_id` the callback contract matches on — and `JobCallbackService`
needs no change at all. `running` stays what it was: the platform worker's own
status, for jobs it executes in process.

The cost is that agent-held pages are governed by the *waiting* sweep rather than
the *running* one, which fails a stale job instead of re-queueing it. Issue 054
owns the lease and is where that is put right; it is recorded here because the
choice is what put the page there.

> **Resolved by issue 054.** ``waiting`` is now two populations, split on the
> ``agent:`` prefix ``claimed_by`` already carried: rows the platform dispatched
> keep the 240-second waiting timeout and still *fail*, and rows an agent holds
> are governed only by the 600-second **lease** and *re-pend*, claim cleared, for
> any agent to take. The split needed no new column, no new state, and no new
> process - the same opportunistic sweep runs it.

### 3. The endpoint takes no request-scoped database session

`Depends(get_db)` pins a pooled connection for the whole request. At
`DB_POOL_SIZE + DB_MAX_OVERFLOW` — fifteen on the defaults — a long poll per idle
agent exhausts the pool, and ADR 0003 puts *all* inference on this path, so the
ceiling binds sooner than it would have for laptops alone.

So the route acquires and releases around each unit of work: authentication opens
and closes a short-lived session before the wait begins, and each claim attempt
opens and closes a sync session inside a worker thread. Between attempts the
request holds no connection, no session, and no row lock — it is asleep.

This is asserted structurally, by walking the route's dependency tree for
`get_db`, with a control assertion that the same walk finds it on a route that
does take one. A load test would only be able to say "it got slow"; the
structural test names the defect.

### 4. A hosted worker's device row is owned by a service account

`helper_devices.user_id` is `NOT NULL` by design (ADR 0001, decision 6): that
foreign key *is* the authorization scope of a device credential, and making it
nullable for the sake of hosted workers would delete the invariant for every
device on the platform. So a hosted worker's row needs an owner.

That owner must not be a researcher. A researcher's account deletion cascades to
their devices, their `GET /devices` would list platform infrastructure, and a
revocation tapped from a phone would stop cloud inference for everyone.

**The owner is a dedicated service account**: one `users` row addressed by a
**fixed UUID5 primary key**, not by email. Keying on the primary key is the
security property — an address can be registered by whoever gets there first, a
fixed uuid5 cannot be — so this can never resolve to an account a person
controls. Its password is a bcrypt hash of a secret discarded on the next line,
so no password grants it; it holds no browser session, no project, and no
document. The only thing it owns is the `inference_host = cloud` rows that report
cloud **capacity**.

Those rows are provisioned on the worker's first claim rather than by hand: a
hosted worker registers itself by working, exactly as a laptop registers itself
by pairing. Provisioning takes an advisory lock, so two workers booting together
produce one row each rather than two rows for one name.

### 5. The agent credentials are added to the existing callback route

Completion and failure are not new endpoints — but a researcher's laptop has no
`INFERENCE_WEBHOOK_SECRET` and must not be given one, because a secret shared
with every agent would let any of them complete any job on the platform.

`POST /internal/inference/job-complete` therefore also accepts the two agent
credentials, narrowed to the page that agent is actually holding: `jobs.claimed_by`
names the claiming device, and nothing else is accepted. A device that merely
*could* have claimed a page cannot report on one it does not hold. An unknown job
answers 403 rather than 404, so the route is not a probe for job ids. The webhook
path is untouched and keeps its 503 / 401 / 403 outcomes exactly.

## Consequences

- **`INFERENCE_WORKER_SERVICE_TOKEN` is not a device token and must not be
  treated like one.** A device token is bounded by one account; this one is
  bounded by nothing, because cloud work belongs to the platform. It is held to
  the same 32-character non-placeholder floor as `DEVICE_TOKEN_HMAC_SECRET`, and
  unset means no hosted worker can claim — which is the correct default while
  none exists.
- **`DEVICE_PAIRING_ENABLED` gates the claim endpoint too**, hosted workers
  included. One switch turns the whole outbound agent layer on, and it is off by
  default in production until the `/pair` page ships.
- **There is no heartbeat.** Work is seconds-to-minutes, the lease covers it with
  margin, and a stopped agent loses one page rather than a document. A unit test
  asserts no heartbeat route exists, so it cannot creep back in without a
  decision.
- **The claim is a poll, not a push.** No SSE, no WebSocket: ADR 0001 records why
  neither survives the production deployment.
