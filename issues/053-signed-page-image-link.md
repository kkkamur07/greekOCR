---
id: "053"
title: "signed-page-image-link"
type: AFK
status: done
tracker: "https://github.com/kkkamur07/greekOCR/issues/53"
blocked_by:
  - "052-device-claim-endpoint.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Deliver the page image to the agent by short-lived signed link carried in the claim response.

The link expires in roughly 60 seconds — long enough to download a large scan on a poor connection, and deliberately **not** tied to the 600-second lease, because the agent fetches once immediately after claiming.

An authenticated image endpoint on the device token was rejected: the production API is serverless, so streaming manuscript scans through it costs money for nothing, and it would place a route on the device token that must independently re-derive job ownership. The signed URL *is* the authorization.

Accepted risk, recorded rather than mitigated: a bearer token in a URL leaks through logs and crash dumps. Bounded to one object and one minute.

## Acceptance criteria

- [ ] The claim response carries a signed link to exactly the one page image for that job
- [ ] The link fetches the real image bytes without any device credential attached
- [ ] The link stops working shortly after its TTL
- [ ] The link grants access to that one object only
- [ ] Its lifetime is independent of the device lease
- [ ] Tested end to end against the real media store, not a stub

## Blocked by

- #52
