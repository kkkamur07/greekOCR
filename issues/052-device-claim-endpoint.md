---
id: "052"
title: "device-claim-endpoint"
type: AFK
status: done
tracker: "https://github.com/kkkamur07/greekOCR/issues/52"
blocked_by:
  - "048-collapse-second-job-queue.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Add the single new endpoint this whole redesign costs: a device-authenticated claim that hands an agent exactly one page of work.

One page per claim. A batch is N claims. Work stays seconds-to-minutes, so the lease covers it with margin and **no heartbeat endpoint is added**; a slept laptop loses one page rather than a document, and progress is free because jobs complete as they go.

Completion and failure post the existing validated job callback contract — they are not new endpoints. Abandonment is caught by the existing stale sweep. Anything the platform already does well is used, not rebuilt.

Two hard constraints:

The endpoint **must not take a request-scoped database session dependency**. A long poll pins a pooled connection for its duration and exhausts the pool at roughly fifteen devices. ADR 0003 moves all inference onto this path, so that ceiling binds sooner than it would have for laptops alone.

A device may only claim work belonging to its own account, and only for the `local` **execution target**. A hosted worker authenticates with a service credential rather than a device token and claims `cloud` work from the same endpoint — one agent implementation, not two code paths kept in parity by discipline. Hosted workers short-poll rather than long-poll; they are never idle for long and do not need the latency.

## Acceptance criteria

- [ ] One claim returns at most one page
- [ ] A device cannot claim another account's work, and cannot claim `cloud` work
- [ ] A service credential can claim `cloud` work from the same endpoint
- [ ] The endpoint holds no pooled connection across a long poll
- [ ] An empty queue returns a well-formed empty response, not an error
- [ ] Completion and failure go through the existing callback contract unchanged
- [ ] Two agents polling concurrently never receive the same page
- [ ] Tested over HTTP against the real application factory with live Postgres

## Blocked by

- #48
