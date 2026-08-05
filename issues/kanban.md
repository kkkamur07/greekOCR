# Kanban

> Regenerated 2026-08-05 — inference redesign ([PRD #47](https://github.com/kkkamur07/greekOCR/issues/47))
>
> Limits: in progress **5**, review **8**, parallel lanes **5**.
>
> Resuming? Read [`docs/resume-inference-redesign.md`](../docs/resume-inference-redesign.md) first.

**All 14 issues are done.** The redesign is complete on `feat/inference-cli-redesign`
and unpushed — `origin/main` still has none of it. What remains is not implementation:
it is the owner's push decision and the credential revocations in §8 of the resume doc.

## Ready (AFK)

The redesign lanes are empty. What is left here was filed *by* the lanes and is
follow-up work, deliberately not folded into the redesign.

- [ ] [062 · bound segmentation peak memory](https://github.com/kkkamur07/greekOCR/issues/62) — found by 049; 7 GB/page will not fit an 8 GB laptop
- [ ] [063 · device pairing event-loop collision](https://github.com/kkkamur07/greekOCR/issues/63) — disables the ADR 0001 route-mounting guard; the 4 red integration tests
- [ ] [064 · announce execution target at enqueue](https://github.com/kkkamur07/greekOCR/issues/64) — found by 059; no host shown until first status update
- [ ] [065 · CUDA pin](https://github.com/kkkamur07/greekOCR/issues/65) — found by 049
- [ ] [066 · rename import package before PyPI](https://github.com/kkkamur07/greekOCR/issues/66) — found by 050; the name is claimable exactly once

## Ready (HITL)

_None as issues, but two owner actions gate the first release_ — revoke eight signing
secrets, and register a PyPI Trusted Publisher plus a `pypi` environment. Resume doc §8.

## In progress

_Empty — 0/5._

## Review

_Empty — 0/8._

## Done

Merged into `feat/inference-cli-redesign`, each as its own `merge:` commit whose body
records the conflicts it resolved. Gates verified, migration chain applies from scratch.

| # | Lane | Shape |
|---|------|-------|
| 048 | [collapse second job queue](048-collapse-second-job-queue.md) | 82 files, +610/−2739 |
| 049 | [Torch runtime, archive ONNX](049-torch-runtime-archive-onnx.md) | output byte-identical, PyTorch faster |
| 050 | [publish inference package](050-publish-inference-package.md) | wheel is `inference/` minus the loopback surfaces |
| 051 | [execution target + capacity](051-execution-target-capacity-gating.md) | migration re-chained to `007` |
| 052 | [device claim endpoint](052-device-claim-endpoint.md) | 25 live acceptance tests; ADR 0005 |
| 053 | [signed page-image link](053-signed-page-image-link.md) | ~60s link, its own dial, not the lease |
| 054 | [device lease stale sweep](054-device-lease-stale-sweep.md) | expired leases re-pend; a closed lid is not a failure |
| 055 | [version floor](055-version-floor.md) | 426 before authentication, so a stale agent stops reporting capacity |
| 056 | [CLI pair + version](056-cli-pair-and-version.md) | 13 live tests: real wheel, real venv, real uvicorn |
| **057** | [**CLI run loop**](057-cli-run-loop.md) | 9 files, +1903/−40; **11 live tests** |
| **058** | [**CLI self-upgrade**](058-cli-self-upgrade.md) | 10 files, +1457/−19; **14 live tests**, real `execve` |
| 059 | [frontend host preference](059-frontend-host-preference.md) | 219 frontend tests |
| **060** | [**delete loopback transport**](060-delete-loopback-transport.md) | 105 files, +1704/−**6714** |
| 061 | [delete native packaging](061-delete-native-packaging.md) | −1954 lines; 8 signing secrets to revoke |

Two side branches merged with the same rule (each worktree's own change wins in its own
domain): `feat/frontend-libraries` (TanStack Query + antd `message` replace hand-rolled
equivalents) and `feat/deep-cleanup` (dead code, generated types, segment numbering, mypy).

## Backlog

_Empty._ 060 was the last card here; it merged on 2026-08-05.

Its load-bearing part is worth recording, because the deadline was invisible: the frontend
published four `releases/latest/download/…` installer URLs and rendered a download button
over them. #061 had already deleted the workflow that builds those artifacts, so the next
release cut would have made `latest` installer-free and every one of those links a 404.
They now point at `uv tool install nomicous-inference` → `nomicous pair` → `nomicous run`,
and the test asserts the panel renders **no links at all**, since the failure mode is a URL
rather than wording.
