# Kanban

> Regenerated 2026-08-04 — inference redesign ([PRD #47](https://github.com/kkkamur07/greekOCR/issues/47))
>
> Limits: in progress **5**, review **8**, parallel lanes **5**.

## Ready (AFK)

- [ ] [062 · bound segmentation peak memory](https://github.com/kkkamur07/greekOCR/issues/62) — found by 049; 7 GB/page will not fit an 8 GB laptop
- [ ] [063 · device pairing event-loop collision](https://github.com/kkkamur07/greekOCR/issues/63) — disables the ADR 0001 route-mounting guard
- [ ] [064 · announce execution target at enqueue](https://github.com/kkkamur07/greekOCR/issues/64) — found by 059; no host shown until first status update

## Ready (HITL)

_None._

## In progress

4/5 — one worktree each, created at merged trunk and base-verified before launch.

- [ ] [050 · publish inference package](050-publish-inference-package.md) — `feat/050-publish-inference-package`
- [ ] [053 · signed page-image link](053-signed-page-image-link.md) — `feat/053-signed-page-image-link`
- [ ] [054 · device lease stale sweep](054-device-lease-stale-sweep.md) — `feat/054-device-lease-stale-sweep`
- [ ] [055 · version floor](055-version-floor.md) — `feat/055-version-floor`

## Review

_Empty — 0/8._

## Done

Merged into `feat/inference-cli-redesign`, gates verified, migration chain applies from scratch.

- [x] [048 · collapse second job queue](048-collapse-second-job-queue.md) — 82 files, +610/−2739
- [x] [049 · Torch runtime, archive ONNX](049-torch-runtime-archive-onnx.md) — output byte-identical, PyTorch faster
- [x] [051 · execution target + capacity](051-execution-target-capacity-gating.md) — migration re-chained to `007`
- [x] [052 · device claim endpoint](052-device-claim-endpoint.md) — 25 live acceptance tests; ADR 0005
- [x] [059 · frontend host preference](059-frontend-host-preference.md) — 219 frontend tests

## Backlog

| Issue | Blocked by |
|-------|-----------|
| [056 · CLI pair + version](056-cli-pair-and-version.md) | 050 |
| [057 · CLI run loop](057-cli-run-loop.md) | 053, 054, 056 |
| [058 · CLI self-upgrade](058-cli-self-upgrade.md) | 055, 056 |
| [060 · delete loopback transport](060-delete-loopback-transport.md) | 057, 059 |
| [061 · delete native packaging](061-delete-native-packaging.md) | 050 |
