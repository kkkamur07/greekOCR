---
id: "056"
title: "cli-pair-and-version"
type: AFK
status: done
tracker: "https://github.com/kkkamur07/greekOCR/issues/56"
blocked_by:
  - "050-publish-inference-package.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

The CLI's first two subcommands: pairing a machine, and reporting its version. Built with `rich`.

Pairing authorises this machine against the researcher's account. **The URL is printed to the terminal first and always**, before any attempt to open a browser — opening a browser is actively wrong over SSH, and the printed URL is the thing that makes the flow work everywhere. Opening the browser is a convenience layered on top, never the only affordance.

The confirmation code is displayed in the terminal and must match what the web page shows. ADR 0001 required this and nothing implemented it; in a CLI it is a `print()`, which is what finally makes the anti-phishing mitigation real rather than decorative.

The device credential is stored under the researcher's home directory with permissions that keep other accounts on the machine from claiming jobs as them.

Pairing an already-paired machine says so rather than silently creating a second device.

## Acceptance criteria

- [x] The pairing URL is printed before any browser is opened
- [x] Pairing completes over SSH with no browser available
- [x] The confirmation code shown in the terminal matches the one on the web page
- [x] The credential is written with owner-only permissions
- [x] Pairing an already-paired machine reports that instead of creating a duplicate device
- [x] A revoked device reports the revocation and exits non-zero rather than spinning
- [x] The version subcommand reports the installed package version
- [x] Tested by running the real CLI against a real running platform with live Postgres

## Blocked by

- #50
