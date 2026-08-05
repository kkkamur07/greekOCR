---
id: "055"
title: "version-floor"
type: AFK
status: in_progress
tracker: "https://github.com/kkkamur07/greekOCR/issues/55"
blocked_by:
  - "052-device-claim-endpoint.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Let the platform refuse agents that are too old.

The platform serves a version floor, and the claim path rejects any agent below it with a response the CLI can act on — distinguishable from a generic error, and carrying enough for the agent to know it must upgrade rather than retry.

The signal comes from the platform rather than PyPI for the same reason every other cadence lives in device settings: it is turnable without a release. It also gives us something frozen installers made impossible — the ability to stop a known-bad agent from claiming work, without waiting for anyone to install anything.

Two distinct states: *below the floor* (refused) and *merely outdated* (served, but told).

## Acceptance criteria

- [ ] The platform serves a version floor that is changeable without a release
- [ ] An agent below the floor is refused with a distinguishable, actionable response
- [ ] An agent at or above the floor claims normally
- [ ] An agent above the floor but behind the latest is served and told it is outdated
- [ ] A missing or malformed agent version is refused rather than assumed current
- [ ] Tested over HTTP against the real application factory

## Blocked by

- #52
