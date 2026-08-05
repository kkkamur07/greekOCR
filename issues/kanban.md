# Kanban

> Regenerated 2026-08-05 — inference redesign ([PRD #47](https://github.com/kkkamur07/greekOCR/issues/47))
>
> Limits: in progress **5**, review **8**, parallel lanes **5**.
>
> Resuming? Read [`docs/resume-inference-redesign.md`](../docs/resume-inference-redesign.md) first.

## Ready (AFK)

Two redesign lanes are open at once for the first time since 052: 057 and 058 both cleared
when 056 merged, and they touch different files.

- [ ] [057 · CLI run loop](057-cli-run-loop.md) — claim, fetch, run, report; every blocker merged
- [ ] [058 · CLI self-upgrade](058-cli-self-upgrade.md) — upgrade when below the version floor
- [ ] [062 · bound segmentation peak memory](https://github.com/kkkamur07/greekOCR/issues/62) — found by 049; 7 GB/page will not fit an 8 GB laptop
- [ ] [063 · device pairing event-loop collision](https://github.com/kkkamur07/greekOCR/issues/63) — disables the ADR 0001 route-mounting guard
- [ ] [064 · announce execution target at enqueue](https://github.com/kkkamur07/greekOCR/issues/64) — found by 059; no host shown until first status update

## Ready (HITL)

_None._

## In progress

_Empty — 0/5._

## Review

_Empty — 0/8._

## Done

Merged into `feat/inference-cli-redesign`, gates verified, migration chain applies from
scratch. Nothing is pushed; `origin/main` has none of it.

- [x] [048 · collapse second job queue](048-collapse-second-job-queue.md) — 82 files, +610/−2739
- [x] [049 · Torch runtime, archive ONNX](049-torch-runtime-archive-onnx.md) — output byte-identical, PyTorch faster
- [x] [051 · execution target + capacity](051-execution-target-capacity-gating.md) — migration re-chained to `007`
- [x] [050 · publish inference package](050-publish-inference-package.md) — wheel is `inference/` minus the loopback surfaces
- [x] [052 · device claim endpoint](052-device-claim-endpoint.md) — 25 live acceptance tests; ADR 0005
- [x] [053 · signed page-image link](053-signed-page-image-link.md) — ~60s link, its own dial, not the lease
- [x] [054 · device lease stale sweep](054-device-lease-stale-sweep.md) — expired leases re-pend; a closed lid is not a failure
- [x] [055 · version floor](055-version-floor.md) — 426 before authentication, so a stale agent stops reporting capacity
- [x] [056 · CLI pair + version](056-cli-pair-and-version.md) — 13 live tests: real wheel, real venv, real uvicorn
- [x] [059 · frontend host preference](059-frontend-host-preference.md) — 219 frontend tests
- [x] [061 · delete native packaging](061-delete-native-packaging.md) — −1954 lines; 8 signing secrets to revoke

Two side branches merged with the same rule (each worktree's own change wins in its own
domain): `feat/frontend-libraries` (TanStack Query + antd `message` replace hand-rolled
equivalents) and `feat/deep-cleanup` (dead code, generated types, segment numbering, mypy).

## Backlog

| Issue | Blocked by |
|-------|-----------|
| [060 · delete loopback transport](060-delete-loopback-transport.md) | 057, ~~059~~ |

060 is the last deletion, and it is load-bearing: the frontend still links to installer
artifacts no workflow produces any more. See §5 of the resume doc.
