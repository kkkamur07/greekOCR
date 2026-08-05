# Kanban

> Regenerated 2026-08-04 — inference redesign ([PRD #47](https://github.com/kkkamur07/greekOCR/issues/47))
>
> Limits: in progress **5**, review **8**, parallel lanes **5** (raised from defaults on owner instruction).

## Ready (AFK)

_Empty — everything unblocked is in progress._

## Ready (HITL)

_None. Every architectural decision was resolved into ADR 0002 / ADR 0003 / ADR 0004 before decomposition._

## In progress

3/5 — one worktree each, created at trunk and base-verified before launch.

- [ ] [048 · collapse second job queue](048-collapse-second-job-queue.md) — `feat/048-queue-collapse`
- [ ] [049 · Torch runtime, archive ONNX](049-torch-runtime-archive-onnx.md) — `feat/049-torch-runtime-archive-onnx`
- [ ] [051 · execution target + capacity](051-execution-target-capacity-gating.md) — `feat/051-execution-target`

> 051 was pulled forward off its DAG blocker. Its dependency on 048 was surface-shrinking,
> not structural; both touch job submission, so both were told to keep changes there
> surgical. It unblocks 052 and 059, the widest part of the graph.

## Review

_Empty — 0/8._

## Done

_Empty._

## Backlog

Blocked on the DAG; promoted automatically as blockers land.

| Issue | Blocked by |
|-------|-----------|
| [050 · publish inference package](050-publish-inference-package.md) | 048, 049 |
| [052 · device claim endpoint](052-device-claim-endpoint.md) | 048 |
| [053 · signed page-image link](053-signed-page-image-link.md) | 052 |
| [054 · device lease stale sweep](054-device-lease-stale-sweep.md) | 052 |
| [055 · version floor](055-version-floor.md) | 052 |
| [056 · CLI pair + version](056-cli-pair-and-version.md) | 050 |
| [057 · CLI run loop](057-cli-run-loop.md) | 053, 054, 056 |
| [058 · CLI self-upgrade](058-cli-self-upgrade.md) | 055, 056 |
| [059 · frontend host preference](059-frontend-host-preference.md) | 051 |
| [060 · delete loopback transport](060-delete-loopback-transport.md) | 057, 059 |
| [061 · delete native packaging](061-delete-native-packaging.md) | 050 |
