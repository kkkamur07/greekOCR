---
id: "060"
title: "delete-loopback-transport"
type: AFK
status: done
tracker: "https://github.com/kkkamur07/greekOCR/issues/60"
blocked_by:
  - "057-cli-run-loop.md 059-frontend-host-preference.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Delete the loopback transport outright — no dual mode, no deprecation window.

This is the slice that removes the fragility the whole redesign is about. A hosted HTTPS page calling `127.0.0.1` depends on a browser permission that Chromium gates behind Private Network Access, that other browsers treat differently, and that any corporate proxy or VPN can break — and when it breaks the researcher cannot tell whether the process is down, the port is taken, the browser blocked it, or their network did.

What goes: the helper's own web server, its CORS allowlist and admission middleware, the frontend discovery, probe, and client layer, the editor's local-run UI, the platform's local inference service, and the loopback entry in the deployed page's content security policy.

Every installed `0.1.6` helper becomes inert. **No migration path is provided** — the install base is negligible and pre-public.

Keeping loopback as a transitional second mode was rejected: every feature — cancel, progress, cache state, error surfaces, job history — would need two implementations permanently, to smooth a cutover for an install base small enough to ignore.

## Acceptance criteria

- [x] Nothing in the repository opens a listening port on the researcher's machine
- [x] The frontend contains no loopback discovery, probe, or client code
- [x] The content security policy no longer permits connections to localhost
- [x] Tests whose subject was the loopback transport are deleted, not ported
- [x] Local inference works end to end with the browser entirely uninvolved in reaching the agent
- [x] Full platform and frontend suites green

## Blocked by

- #57
- #59
