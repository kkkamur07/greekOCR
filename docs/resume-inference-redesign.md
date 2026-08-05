# The inference redesign — complete, unpushed

**Start here.** As of 2026-08-05 all fourteen issues under PRD
[#47](https://github.com/kkkamur07/greekOCR/issues/47) are implemented, merged and verified
on `feat/inference-cli-redesign`. **Nothing has been pushed.** `origin/main` has none of it.

What is left is not implementation. It is the owner's push decision (§7) and two credential
actions (§8).

---

## 1. Where the work is

| | |
|---|---|
| Trunk branch | `feat/inference-cli-redesign` |
| Head | `git log -1` — the last `docs:` commit of 2026-08-05 |
| Ahead of `origin/main` | ~100 commits, **none pushed** |
| Worktrees | none — all 17 removed after verifying each branch is an ancestor of the trunk |
| Parent PRD | [#47](https://github.com/kkkamur07/greekOCR/issues/47) |
| Decisions | ADRs [0001](./adr/0001-outbound-helper-device-pairing.md)–[0005](./adr/0005-agent-claim-endpoint-and-the-inference-service-account.md) |
| Glossary | [`inference/CONTEXT.md`](../inference/CONTEXT.md) |

Every lane branch (`feat/048-…` … `feat/061-…`, plus `feat/deep-cleanup` and
`feat/frontend-libraries`) is kept as the audit trail. `git branch --no-merged HEAD` is empty.
[`merge-handoff-inference-redesign.md`](./merge-handoff-inference-redesign.md) is history now
— read it only for *why* a particular conflict was resolved the way it was.

## 2. What was built

All 14 merged, each as its own `merge:` commit whose body records the conflicts it resolved.
See [`issues/kanban.md`](../issues/kanban.md) for the per-lane table.

The shape of the change: a browser talked to a loopback HTTP server on the researcher's own
machine; now a CLI installed from PyPI pairs with the platform and **pulls** work outbound.

- **048–052** collapsed the second job queue, moved the runtime to PyTorch (ADR 0004),
  published the package, added the execution target, and opened the device claim endpoint.
- **053–056** added the signed page-image link, the device lease and stale sweep, the version
  floor, and `nomicous pair` / `nomicous version`.
- **057** closed the loop: `nomicous run` claims a page, fetches it through the signed link,
  runs the model, reports through the existing job callback. 11 live tests.
- **058** made the launch moment the only point at which the agent replaces its own code.
  14 live tests, real `execve`, a real local PEP 503 index.
- **059, 061** the frontend host preference, and the deletion of native packaging.
- **060** deleted the loopback transport itself: `inference/api`, `inference/helper`, the
  platform's local-inference persist routes, the frontend probe/client layer, and the
  `127.0.0.1:8001` CSP entry. −6714 lines.

**The published package now contains no code path that opens a listening socket.** That is
ADR 0002 made structural rather than documented, and it is worth re-checking after any merge:

```bash
grep -rnE "uvicorn\.run|HTTPServer|\.listen\(|socket\.bind|\.bind\(" inference/ --include='*.py'   # must be empty
```

Schema is at alembic head `007_execution_target`.

## 3. What red looks like, so a real regression is visible

Run the suites **separately** — they share one Postgres, and running them together is what
produces phantom failures (see the notes below).

| Command | Expected |
|---|---|
| `pytest tests/nomicous` | **668 passed, 1 skipped, 0 failed** |
| `pytest tests/inference tests/hf` | **207 passed, 2 skipped, 0 failed** |
| `cd nomicous/frontend && npm test` | 45 files, 182 passed |
| `npx tsc --noEmit` / `npx eslint .` | clean / 0 errors, 2 warnings |
| `ruff check .` | **All checks passed** |
| `ruff format --check` (CI's path list) | clean; `src/` is excluded from both ruff gates |

**Green means green.** Any red is now a signal. The suite used to carry nine documented
platform failures; all nine were real defects, and none was in the code the tests pointed at:

- **4 × `integration/test_device_pairing.py`** — the module assembled its own app and opened
  a second `TestClient`, so its queries ran on a second event loop while the asyncpg pool
  belonged to the first. It did that only to set the poll cadence before `backend.core.app`
  imports; the env var moved to the integration conftest, which loads earlier still.
- **4 × empty `caplog`** — `alembic/env.py` called `fileConfig(path)`, which defaults to
  `disable_existing_loggers=True` and switched off every existing logger for the rest of the
  session. Not only a test bug: any process that configured logging and then ran a migration
  lost its logging silently.
- **1 × poll cadence** — `DeviceSettings` is a pydantic-settings model, so anything not passed
  explicitly comes from the environment, and the unit fixture inherited the integration
  conftest's `DEVICE_PAIRING_POLL_INTERVAL_SECONDS=1`. The fixture now states what it asserts.
- **1 × `test_process_one_job_runs_every_stale_sweep`** — patched three of the worker's four
  sweeps, so the lease sweep opened a live Postgres connection from a *unit* test. It passed
  only when an integration test had provisioned a database earlier in the same session.
- **1 × `test_grayscale_helper_is_the_only_convention_under_src`** — allow-listed, with the
  reason: `weather.py`'s `COLOR_RGB2GRAY` is a luminance term for a snow blend, not the
  train/serve conversion. **Audit finding recorded there and not fixed**, because `src/` is
  audit-only: that same function does its real grayscale with PIL's `ImageOps.grayscale`, and
  PIL and OpenCV use different luma coefficients — the exact skew that module exists to
  prevent, by a route its marker list cannot see.

`src/` is excluded from ruff wholesale via `extend-exclude`. That is a **suppression**, not a
fix: pointing ruff at `src/` directly still reports its ~576 findings. The previous config
listed `src/model` (singular) and silently missed `src/models` (plural), which was 159 of the
164 findings — a whole-tree exclusion cannot drift that way.

> **A tenth failure that is not real.** Running another suite, or another agent, against the
> same Postgres concurrently produces one extra failure — and **not always the same one**. It
> was `test_device_lease.py::test_concurrent_agents_racing_a_swept_queue…` in one run and
> `test_documents.py::test_upload_reorder_delete_part_and_serve_media` in the next; both pass
> together in isolation. A regression does not move. Measure on an idle database before
> believing a tenth red.

> **A whole suite of errors that is also not real.** If ~146 tests ERROR at fixture setup and
> the run stops around 520 passed, the database is gone, not the code. Check
> `docker ps --filter name=nomicous-db-1` first.

## 4. Getting an environment that works

```bash
# Postgres — container nomicous-db-1, exposed on 127.0.0.1:5433
docker ps --filter name=nomicous-db-1 || docker start nomicous-db-1

# Python. Bare `uv sync` PRUNES the venv to the default groups and takes zxcvbn,
# fastapi and the rest with it; every backend import then fails. Always name the groups.
# NOTE: `--group helper` is gone as of #060.
uv sync --group dev --group test --group platform --group inference

# Frontend
cd nomicous/frontend && npm install
```

Tests are live by owner's instruction: real Postgres, real `create_app()`, real console
scripts, real wheels in real venvs. If something cannot be tested live, defer it rather than
mocking it — installing a package into an isolated environment to make it live is allowed.
`test_cli_pairing.py`, `test_cli_run.py` and `test_cli_self_upgrade.py` are the models.

## 5. Traps that have already cost time

1. **A merge can produce code that parses and is still dead.** Resolving a conflict by taking
   both sides in file order put a method-indented `def read_agent_floor` *after* the
   module-level helpers, where Python silently absorbed it as a nested function. The file
   imported, `ast` parsed it, and `PlatformClient.read_agent_floor` did not exist. Verify a
   merged class by asserting the attribute (`hasattr`), never by checking that the file
   imports.
2. **Every claim must send `X-Nomicous-Agent-Version`.** #055 made it mandatory and evaluates
   it *before* authentication (HTTP 426). A client written against a pre-055 tree gets refused
   with a status nobody expects.
3. **Settings read an ambient dotenv.** `backend/core/settings/_env.py` resolves
   `backend/core/.env`, falling back to `.env.supabase` — both gitignored. A test that assumes
   a backend must pin it (`monkeypatch.setenv("STORAGE_BACKEND", "local")` plus
   `reset_settings_caches()`). Before that fix, `test_signed_page_image_link.py` was uploading
   manuscript pages to a **live Supabase project** in any checkout that had the file.
4. **Never memoize anything settings-derived with a bare `lru_cache`.** Use `@settings_cache`
   from `backend/core/settings/_cache.py` so `reset_settings_caches()` can reach it.
5. **Do not capture a settings-derived object in `__init__`.**
6. **ADR 0004 is invisible in a diff.** A three-way merge resolves "modified here, deleted
   there" by keeping the modification, silently resurrecting ONNX. After any merge:
   `grep -rl onnxruntime inference/` must be empty and `inference/architectures/*/onnx.py`
   must not exist.
7. **`src/` is audit-only.** Report what is wrong there; do not edit it.

## 6. Known gaps, deliberately not closed

- **`selectedModelHostEligibility`** ("this model is remote-only") is gone from the settings
  panel. Its only data source was the loopback probe; the platform's `/inference/models`
  publishes no `host_eligibility`. Restoring it needs a platform change.
- **`GET /inference/v1/registry`** survives though its only consumer was deleted. It is
  platform surface, not transport.
- **The device API still says "helper"** (`helper_version` column, pairing docstrings) and
  that word leaks into the generated `schema.d.ts`. Renaming needs a migration.
- **`ClaimedPageResponse.request` carries the page image twice** — inline base64 *and*
  `page_image_url` — which contradicts ADR 0002's cost rationale. Reported, not fixed;
  touches `tests/nomicous/integration/test_device_claim.py`.
- **Adding a model is now a package release, not a hot registry sync.** The CLI resolves
  against the `registry.yaml` inside its own wheel; `HELPER_REGISTRY_URL` lived only in the
  deleted helper. Documented in `docs/inference/adding-inference-models.md`.
- Follow-ups filed by the lanes themselves: [#62](https://github.com/kkkamur07/greekOCR/issues/62)
  (segmentation peak memory), [#63](https://github.com/kkkamur07/greekOCR/issues/63),
  [#64](https://github.com/kkkamur07/greekOCR/issues/64),
  [#65](https://github.com/kkkamur07/greekOCR/issues/65) (CUDA pin),
  [#66](https://github.com/kkkamur07/greekOCR/issues/66) (**rename the import package before
  PyPI** — the name is claimable exactly once).

## 7. The push — the owner's call

The trunk has never been pushed. Whether these ~100 commits reach `main` as a merge, a
rebase, or a pull request has not been decided, and no agent should decide it.

## 8. Owner's actions — not an agent's

### Revoke eight CI secrets

#061 deleted the native installer pipeline, so these are unused credentials **with signing
authority**. Deleting the repository secret is only half of it; the parenthesised half
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
`signtool`, `notarytool`, or any `SIGNING` variable any more. A published Python package is
not signed by its author: PyPI Trusted Publishing establishes provenance through OIDC, and
`actions/attest-build-provenance` records it under PEP 740. There is no key to hold.

### Before the first release can be cut

`.github/workflows/release.yml` (added by #061, triggers on `v*` tags) publishes with
`uv publish --trusted-publishing always` and holds no secrets. It needs, one time:

1. a **PyPI Trusted Publisher** registered for this repository and that workflow filename;
2. a GitHub environment named **`pypi`**;
3. issue [#66](https://github.com/kkkamur07/greekOCR/issues/66) resolved first — the package
   name is claimable exactly once.
