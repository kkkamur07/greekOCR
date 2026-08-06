# Kanban

> Regenerated 2026-08-05 for the inference redesign ([PRD #47](https://github.com/kkkamur07/greekOCR/issues/47)).
>
> Limits: in progress 5, review 8, parallel lanes 5.
>
> Resuming? Read [`docs/resume-inference-redesign.md`](../docs/resume-inference-redesign.md) first.

All 14 issues are done. The redesign is complete on `feat/inference-cli-redesign` and
unpushed, so `origin/main` still has none of it. What remains is not implementation: it is
the owner's push decision and the credential revocations in §8 of the resume doc.

## Ready (AFK)

The redesign lanes are empty. What is left here was filed by the lanes themselves and is
follow-up work, deliberately kept out of the redesign.

- [ ] [062 · bound segmentation peak memory](https://github.com/kkkamur07/greekOCR/issues/62). Found by 049. 7 GB/page will not fit an 8 GB laptop.
- [ ] [063 · device pairing event-loop collision](https://github.com/kkkamur07/greekOCR/issues/63). Disables the ADR 0001 route-mounting guard, and accounts for the 4 red integration tests.
- [ ] [064 · announce execution target at enqueue](https://github.com/kkkamur07/greekOCR/issues/64). Found by 059. No host is shown until the first status update.
- [ ] [065 · CUDA pin](https://github.com/kkkamur07/greekOCR/issues/65). Found by 049.
- [ ] [066 · rename import package before PyPI](https://github.com/kkkamur07/greekOCR/issues/66). Found by 050. The name is claimable exactly once.

## Ready (HITL)

No issues, but two owner actions gate the first release: revoke eight signing secrets, and
register a PyPI Trusted Publisher plus a `pypi` environment. See resume doc §8.

## In progress

Empty, 0/5.

## Review

Empty, 0/8.

## Done

Each lane merged into `feat/inference-cli-redesign` as its own `merge:` commit, whose body
records the conflicts it resolved. Gates verified, and the migration chain applies from
scratch. The issue files themselves have been deleted now that the work has shipped; the
merge commits are the surviving record.

| #   | Lane                          | Shape                                                                      |
| --- | ----------------------------- | -------------------------------------------------------------------------- |
| 048 | collapse second job queue     | 82 files, +610/−2739                                                       |
| 049 | Torch runtime, archive ONNX   | output byte-identical, PyTorch faster - reversed by ADR 0006 on closure size |
| 050 | publish inference package     | wheel is `inference/` minus the loopback surfaces                          |
| 051 | execution target + capacity   | migration re-chained to `007`                                              |
| 052 | device claim endpoint         | 25 live acceptance tests; ADR 0005                                         |
| 053 | signed page-image link        | roughly 60s link, on its own dial rather than the lease                    |
| 054 | device lease stale sweep      | expired leases re-pend, because a closed lid is not a failure              |
| 055 | version floor                 | 426 before authentication, so a stale agent stops reporting capacity       |
| 056 | CLI pair + version            | 13 live tests against a real wheel, a real venv, and real uvicorn          |
| 057 | CLI run loop                  | 9 files, +1903/−40; 11 live tests                                          |
| 058 | CLI self-upgrade              | 10 files, +1457/−19; 14 live tests, real `execve`                          |
| 059 | frontend host preference      | 219 frontend tests                                                         |
| 060 | delete loopback transport     | 105 files, +1704/−6714                                                     |
| 061 | delete native packaging       | −1954 lines; 8 signing secrets to revoke                                   |

Two side branches merged under the same rule, that each worktree's own change wins in its
own domain: `feat/frontend-libraries` (TanStack Query and antd `message` replace hand-rolled
equivalents) and `feat/deep-cleanup` (dead code, generated types, segment numbering, mypy).

## Backlog

Empty. 060 was the last card here, and it merged on 2026-08-05.

Its load-bearing part is worth recording, because the deadline was invisible. The frontend
published four `releases/latest/download/…` installer URLs and rendered a download button
over them. #061 had already deleted the workflow that builds those artifacts, so the next
release cut would have made `latest` installer-free and every one of those links a 404. They
now point at `uv tool install nomicous-inference`, then `nomicous pair`, then `nomicous run`,
and the test asserts the panel renders no links at all, since the failure mode is a URL
rather than wording.
