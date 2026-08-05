# Resuming the inference redesign

**Start here.** This is the state of the work as of 2026-08-05, written so it can be picked
up cold. [`merge-handoff-inference-redesign.md`](./merge-handoff-inference-redesign.md) is
now history: the merge it describes has been carried out, and every branch it lists is in
the trunk. Read it only for *why* a conflict was resolved the way it was.

Three of fourteen issues remain. Nothing has been pushed.

---

## 1. Where the work is

| | |
|---|---|
| Trunk branch | `feat/inference-cli-redesign` |
| Head | the last `docs:` commit of 2026-08-05 — `git log -1` |
| Ahead of `origin/main` | ~85 commits, **none pushed** |
| Working tree | clean |
| Parent PRD | [#47](https://github.com/kkkamur07/greekOCR/issues/47) |
| Decisions | ADRs [0001](./adr/0001-outbound-helper-device-pairing.md)–[0005](./adr/0005-agent-claim-endpoint-and-the-inference-service-account.md) |
| Glossary | [`inference/CONTEXT.md`](../inference/CONTEXT.md) — the vocabulary every issue and commit message uses |

`main` and `origin/main` are untouched by any of this. The redesign exists only on the
trunk branch and in the worktrees listed in §7.

## 2. What is done

Eleven lanes are merged, each as its own `merge:` commit whose body records the conflicts
it resolved.

| # | Lane | State |
|---|---|---|
| 048 | collapse the second job queue | merged |
| 049 | Torch runtime, ONNX archived | merged |
| 050 | publish the inference package | merged |
| 051 | execution target + capacity gating | merged |
| 052 | device claim endpoint | merged |
| 053 | signed page image link | merged |
| 054 | device lease + stale sweep | merged |
| 055 | version floor | merged |
| 056 | `nomicous pair` and `nomicous version` | merged, **13 live tests passing** |
| 059 | frontend host preference | merged |
| 061 | delete native packaging; release is a PyPI publish | merged |
| — | `feat/deep-cleanup` | merged |
| — | `feat/frontend-libraries` | merged |
| **057** | **CLI run loop** | **not started** |
| **058** | **CLI self-upgrade** | **not started** |
| **060** | **delete the loopback transport** | **not started** |

Schema is at alembic head `007_execution_target`.

## 3. What red looks like, so a real regression is visible

Run the suites separately — the numbers below are per-suite, and the last row explains why.

| Command | Expected |
|---|---|
| `uv run pytest tests/nomicous` | 669 passed, 1 skipped, **9 failed** |
| `uv run pytest tests/inference tests/hf` | 1 failed (`test_calamari_grayscale_parity`) |
| `cd nomicous/frontend && npm test` | 51 files, 217 passed |
| `npx tsc --noEmit` / `npx eslint .` | clean / 0 errors, 2 warnings |
| `uv run --group dev ruff check .` | **164 errors, all under `src/`** |

The failures are all pre-existing and all understood:

- **4 × `tests/nomicous/integration/test_device_pairing.py`** — asyncpg event-loop
  collision, issue #63. Not a product bug.
- **5 × caplog cross-contamination** (`unit/test_device_pairing.py` ×4,
  `unit/test_job_callback_service.py` ×1). Passing `-p no:logging` turns these five into
  ERRORs instead, so only compare like-for-like invocations.
- **1 × `test_grayscale_helper_is_the_only_convention_under_src`** —
  `src/models/trocr/augmentation/weather.py` uses `COLOR_RGB2GRAY` and is not in the test's
  allow-list. It is under `src/`, which is audit-only, so it is reported rather than fixed.
- **164 ruff errors** — every one under `src/`. The `per-file-ignores` block names
  `src/model/**` (singular); the offending tree is `src/models/**` (plural), restored later
  by `516c3fc`, so the ignores never matched it. Again `src/`, again audit-only. Fixing it
  is a config change plus a decision about the vendored tree, not a merge task.

One further failure appears **only in a single whole-suite `uv run pytest` run**:
`test_device_lease.py::test_concurrent_agents_racing_a_swept_queue_each_get_a_distinct_page`.
It passes alone, passes with its own file, passes across `tests/nomicous`, and passes
across `tests/nomicous/integration tests/inference/integration`. It is an ordering
interaction, not a lane regression — `tests/nomicous` produces byte-identical results
before and after every merge in §2. Track it with the other two contamination families
rather than treating a whole-suite red as a blocker.

## 4. Getting an environment that works

```bash
# Postgres — container nomicous-db-1, already exposed on 127.0.0.1:5433
docker ps --filter name=nomicous-db-1

# Python. Bare `uv sync` PRUNES the venv to the default groups and takes zxcvbn,
# fastapi and the rest with it; every backend import then fails. Always name the groups:
uv sync --group dev --group test --group platform --group inference

# Frontend
cd nomicous/frontend && npm install
```

Tests are live by owner's instruction: real Postgres, real `create_app()`, real console
scripts. If something cannot be tested live, defer it rather than mocking it — installing a
package into an isolated environment to make it live is allowed. `test_cli_pairing.py`
is the model: a real wheel in its own venv, a real uvicorn process, its own database
(`kalamos_056_cli`), and approval driven by a second process over HTTP.

## 5. What is left

### #057 — CLI run loop *(blocked by 053, 054, 056 — all merged, so it is ready)*

The `nomicous run` subcommand: claim a page, fetch it through the **signed page image
link**, run the model, report through the platform's existing job callback, repeat.
Everything it needs is on the trunk — the claim endpoint (052), the link (053), the lease
(054), and the credential and version header (056, 055).

### #058 — CLI self-upgrade *(blocked by 055, 056 — ready)*

Reads the **version floor** the platform advertises and upgrades the installed package
when it falls below it. `nomicous version` already reports what the floor will read.

### #060 — delete the loopback transport *(blocked by 057, 059)*

The last deletion. It removes `inference/api` and `inference/helper` (both are already
excluded from the wheel — see `[tool.hatch.build.targets.wheel] exclude` in
`pyproject.toml`) and the frontend's helper client.

> **Do not skip #060 indefinitely.** `nomicous/frontend/src/inference/constants.ts` still
> publishes four installer download URLs pointing at
> `github.com/kkkamur07/greekOCR/releases/latest/download/…`, and
> `PageEditorInferenceBanner.tsx` still renders a "Download the installer" button over
> them. They work today only because the old `v0.1.6` release still exists. #061 deleted
> the workflow that produces those artifacts, so the first release cut after this lands
> makes `latest` an installer-free release and every one of those links 404s.

## 6. Traps that have already cost time

1. **Every claim must send `X-Nomicous-Agent-Version`.** #055 made it mandatory and
   evaluates it *before* authentication (HTTP 426). Any test helper or client written
   against a pre-055 tree gets refused with a status nobody expects.
2. **Settings read an ambient dotenv.** `backend/core/settings/_env.py` resolves
   `backend/core/.env`, falling back to `.env.supabase` — both gitignored. A test that
   assumes a backend must pin it (`monkeypatch.setenv("STORAGE_BACKEND", "local")` plus
   `reset_settings_caches()`). Before that fix, `test_signed_page_image_link.py` was
   uploading manuscript pages to a **live Supabase project** in any checkout that had the
   file.
3. **Never memoize anything settings-derived with a bare `lru_cache`.** Use
   `@settings_cache` from `backend/core/settings/_cache.py` so `reset_settings_caches()`
   can reach it. `get_media_store` was the leak.
4. **Do not capture a settings-derived object in `__init__`.** Four route modules build
   `DocumentPartService()` at import time; it now resolves the media store through a
   property per use for exactly that reason.
5. **ADR 0004 is invisible in a diff.** PyTorch replaced ONNX Runtime and the ONNX code
   moved to `archive/onnx-runtime/`. A three-way merge resolves "modified here, deleted
   there" by keeping the modification, which silently resurrects it. After any merge:
   `grep -rl onnxruntime inference/` must be empty and `inference/architectures/*/onnx.py`
   must not exist.
6. **`src/` is audit-only.** Report what is wrong there; do not edit it.

## 7. Worktrees still on disk

All are merged and safe to remove.

```
$WT/wt-{048,050,051,052,053,054,055,056,059,061}
   where $WT=/private/tmp/claude-501/-Users-krishuagarwal-Desktop-Programming-python-greekOCR/b5e3d048-1766-4578-a155-c909851d5110
/Users/krishuagarwal/Desktop/Programming/python/greekOCR-deepclean      feat/deep-cleanup
/Users/krishuagarwal/Desktop/Programming/python/greekOCR-frontend       feat/frontend-libraries
/Users/krishuagarwal/Desktop/Programming/python/greekOCR/.claude/worktrees/agent-af7f62c472b9cf6ac
```

`git worktree list` is authoritative; `git worktree remove <path>` then
`git branch -d <branch>` once you are satisfied the trunk carries everything.

## 8. Owner's actions — not an agent's

### Revoke eight CI secrets

#061 deleted the native installer pipeline, so these are unused credentials with signing
authority. Deleting the repository secret is only half of it; the parenthesised half
revokes the underlying credential.

| Secret | Also revoke |
|---|---|
| `MACOS_CERTIFICATE_P12` | the Developer ID certificate, at Apple |
| `MACOS_CERTIFICATE_PASSWORD` | — |
| `MACOS_CODESIGN_IDENTITY` | — |
| `MACOS_NOTARY_PROFILE` | the app-specific password |
| `WINDOWS_SIGNING_CERT_BASE64` | the Authenticode certificate, at the CA |
| `WINDOWS_SIGNING_CERT_PASSWORD` | — |
| `RELEASE_SIGNING_GPG_KEY` | publish a revocation certificate |
| `RELEASE_SIGNING_GPG_PASSPHRASE` | — |

Nothing in `.github/workflows/` or `deploy/` references `cosign`, `gpg`, `codesign`,
`signtool`, `notarytool`, or any `SIGNING` variable any more. A published Python package
is not signed by its author: PyPI Trusted Publishing establishes provenance through OIDC,
and `actions/attest-build-provenance` records it under PEP 740. There is no key to hold.

### Before the first release can be cut

`.github/workflows/release.yml` (added by #061, triggers on `v*` tags) publishes with
`uv publish --trusted-publishing always` and holds no secrets. It needs, one time:

1. a **PyPI Trusted Publisher** registered for this repository and that workflow filename;
2. a GitHub environment named **`pypi`**.

### The push

The trunk has never been pushed. Whether these commits reach `main` as a merge, a
rebase, or a pull request is the owner's call.

## 9. Open questions

- Keep `release.yml` inside #061, or split it into its own issue? It is separable at
  commit `131d6be`.
- The `src/` lint and grayscale findings (§3) need a decision: fix the `src/model` vs
  `src/models` mismatch in `per-file-ignores`, exclude the tree outright, or leave both
  gates red on purpose.
