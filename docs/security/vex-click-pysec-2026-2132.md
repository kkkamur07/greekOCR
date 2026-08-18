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

The lockfile resolves Click 8.2.1, below the 8.3.3 fix floor, through two
independent transitive paths, not "CLI tooling" generically:

- `typer`, pulled in by `huggingface-hub` (a direct `[project.dependencies]`
  entry, so it resolves in every group); `typer` depends on `click`.
- `uvicorn[standard]`, which depends on `click` directly and is itself in the
  shipping `inference` and `platform` groups.

The original Kraken package is gone from the repository entirely: it was the
oracle for the ONNX parity harness, and ADR 0004 retired both, but Click did
not leave with it - the two paths above keep it resolved regardless.

## Reachability

- `grep -rn "click.edit"` across `nomikos/` and `src/` finds no callers:
  nothing in this codebase invokes `click.edit()`.
- Product and inference runtime entrypoints (`uvicorn` apps, job workers, helper
  `/run`) do not call it either - `uvicorn[standard]` uses Click only for its
  own CLI, not as something our request-handling code calls into.
- Click is present only as a transitive dependency of `typer` and `uvicorn`, not
  as an application API that accepts untrusted filenames into `click.edit()`.

## Mitigation / next step

Revisit when the inference dependency graph resolves Click `>=8.3.3`, then remove
the `pip-audit` ignore and delete this VEX note.
