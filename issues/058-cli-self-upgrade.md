---
id: "058"
title: "cli-self-upgrade"
type: AFK
status: backlog
tracker: "https://github.com/kkkamur07/greekOCR/issues/58"
blocked_by:
  - "055-version-floor.md 056-cli-pair-and-version.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Self-upgrade at launch, and only at launch.

A CLI has something a daemon does not: a launch moment with no in-flight work. The agent asks the platform for its version floor on start, upgrades itself and re-execs if it is below the floor, prints a notice when it is merely outdated, and only then begins claiming.

**Never mid-session.** A process that swaps its own code during a batch is a bug generator.

A failed upgrade is loud and fatal — a clear message and a non-zero exit — because the failure mode to avoid is a researcher believing work is happening when it is not.

Accepted risk, recorded rather than mitigated: auto-upgrade executes newly fetched code without asking, so a compromised package reaches every researcher's laptop at next launch. Mitigable by pinning to published hashes; not eliminable. A notice telling users to upgrade themselves was rejected as ignorable, and stale agents are exactly the population that ignores notices.

## Acceptance criteria

- [ ] An agent below the floor upgrades, re-execs, and then claims
- [ ] An agent merely outdated prints a notice and claims without upgrading
- [ ] An agent at the current version prints nothing and claims
- [ ] No upgrade is ever attempted after claiming has begun
- [ ] A failed upgrade prints a clear message and exits non-zero without claiming
- [ ] Exercised against a real local package index; if that proves impractical, the path is deferred and recorded as untested rather than mocked

## Blocked by

- #55
- #56
