---
id: "058"
title: "cli-self-upgrade"
type: AFK
status: done
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

- [x] An agent below the floor upgrades, re-execs, and then claims
- [x] An agent merely outdated prints a notice and claims without upgrading
- [x] An agent at the current version prints nothing and claims
- [x] No upgrade is ever attempted after claiming has begun
- [x] A failed upgrade prints a clear message and exits non-zero without claiming
- [x] Exercised against a real local package index; if that proves impractical, the path is deferred and recorded as untested rather than mocked

## As built

The launch check is `inference/cli/upgrade.py`, wired into `main.py` for the
commands that claim (`_CLAIMS_WORK`) and nowhere else. It is also the whole of
`nomicous upgrade`, which runs the same code path with the same output — down to
printing nothing when the agent is current — so the behaviour is observable
before #57's run loop exists.

"Asks the platform for its version floor" needed somewhere to ask that is not
the claim endpoint: an agent that had to claim to learn it was stale would be
holding a page at the moment it replaced its own code. So the platform now also
serves `GET /device/v1/agent/version` — the same comparison, the same 426, the
same notice, and nothing taken from the queue. Unauthenticated, because the
version dependency already resolves before any credential is looked at, and it
discloses nothing a 426 does not.

The upgrade runs whichever installer already owns the environment: `pip` when
this interpreter has one, `uv pip install --python sys.executable` when it does
not (a `uv tool install` environment has no pip in it). Then `os.execve` into the
same argument vector, carrying `NOMICOUS_UPGRADED_FROM` — whose presence is the
loop guard: a build that upgraded and is *still* below the floor stops with a
fatal error naming both versions rather than fetching the same wheel forever.

Accepted risk, recorded in `inference/cli/upgrade.py`, ADR 0002 and
`inference/CONTEXT.md`: auto-upgrade executes newly fetched code without asking,
so a compromised package reaches every laptop at next launch. Mitigable by
pinning to published hashes — not done — and not eliminable. Narrowed but not
closed by the platform naming a package rather than a command, and by the index
being the researcher's configured one rather than one the platform picked.

### Notes on the ticks

- *"and then claims"* is proved at the platform: the same real device credential
  is refused `426` at `0.1.0` and served `200` at `0.9.0` after the CLI has
  upgraded itself. The claim is issued by the test rather than by the CLI,
  because the claim loop is #57; there is nothing left for #57 to do here beyond
  existing, since the gate returns before the command runs.
- *"no upgrade after claiming has begun"* is structural — one call site, in
  `main()`, before dispatch — and evidenced live by counting requests to the
  floor endpoint in the platform's own access log: exactly one per launch, and
  none at all from `version` or `--help`. The mid-batch case becomes directly
  observable once #57's loop exists; it has no call site to upgrade from.

Tests: `tests/inference/integration/test_cli_self_upgrade.py` (14, all live).
Two agent builds are compiled from this repository's `inference/` tree into a
PEP 503 index served over HTTP by `python -m http.server`, and the real `pip`
and real `uv` resolve against it. The test wheels declare no dependencies so the
resolver never leaves the local index; whether the published closure resolves is
`test_published_package.py`'s question.

## Blocked by

- #55
- #56
