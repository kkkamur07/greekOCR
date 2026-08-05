# VEX: PYSEC-2026-2132 / CVE-2026-7246 (Click)

**Status:** not_affected (not reachable in deployed product paths)
**Owner:** platform
**Created:** 2026-07-13
**Review by:** 2026-10-13
**Ignore site:** `.github/workflows/security.yml` (`pip-audit --ignore-vuln PYSEC-2026-2132`)

## Vulnerability

Click versions before 8.3.3 allow command injection via `click.edit()` when an
attacker controls the filename argument (`shell=True` command construction).

## Why the ignore remains

The current inference dependency graph resolves Click 8.2.1 through CLI tooling.
The original Kraken package is gone from the repository entirely: it was the
oracle for the ONNX parity harness, and ADR 0004 retired both.

## Reachability

- Product and inference runtime entrypoints (`uvicorn` apps, job workers, helper
  `/run`) do not call `click.edit()`.
- Click is present only as a transitive CLI dependency, not as an application API
  that accepts untrusted filenames into `click.edit()`.

## Mitigation / next step

Revisit when the inference dependency graph resolves Click `>=8.3.3`, then remove
the `pip-audit` ignore and delete this VEX note.
