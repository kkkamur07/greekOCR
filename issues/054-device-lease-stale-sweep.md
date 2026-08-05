---
id: "054"
title: "device-lease-stale-sweep"
type: AFK
status: backlog
tracker: "https://github.com/kkkamur07/greekOCR/issues/54"
blocked_by:
  - "052-device-claim-endpoint.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Release claims from agents that stopped, using the stale sweep the platform already has rather than a new mechanism.

The device lease is 600 seconds. The global 1800-second job timeout is right for a server that does not sleep and wrong for a laptop that does — a closed lid should not hold a page for half an hour.

A crash, a closed laptop, or a killed process releases the page back to the queue once the lease expires. Because the sweep is opportunistic and runs on read paths, no background worker is introduced; the deployment is serverless and must stay that way.

A hosted worker inherits the same lease semantics. A server that does not sleep will never trip it, but it is now the one timeout rather than one of two.

## Acceptance criteria

- [ ] A claimed page whose lease has expired returns to the queue and can be claimed again
- [ ] A page within its lease is not swept
- [ ] The sweep runs opportunistically with no background process added
- [ ] Concurrent sweeps do not double-release or corrupt job state
- [ ] The device lease is distinct from, and shorter than, the global job timeout
- [ ] Tested against live Postgres through the real job lifecycle

## Blocked by

- #52
