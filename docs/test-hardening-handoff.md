# Test hardening — handoff (branch `test/suite-hardening`)

Plan: `~/.claude/plans/binary-growing-donut.md`. Findings: `docs/test-suite-review-2026-08-05.md`.

## Done
- **A — frontend auth un-mocked.** 6 files. `npm test` 45 files / 200 tests green, lint clean.
  Mutation-verified: no-op `establish`+`logout` → 3 red; drop `redirectToLogin` → 2 red;
  dead router → 2 red; drop the `loginRedirectInFlight` guard → 2 red; drop the
  https/localhost rule → 1 red; sign-in storing no token / swallowing the error → 1 red each.
- **B2 — unit sweep ordering** (`tests/nomicous/unit/test_job_lifecycle.py`): all five sweeps
  patched, no DB touched from the no-DB CI lane.
- **B1 — `tests/nomicous/integration/test_job_worker_sweeps.py`** (new, 6 tests, live Postgres).
  Mutation-verified 9/9: each of the four sweep calls removed from `process_one_job`; the agent
  exclusion dropped from the waiting timeout (the ADR 0005 regression); the reclaim deadline; the
  lease deadline; `updated_at=Job.updated_at` in the claim release; the claim clearing in the
  lease sweep. Production restored after each — `git diff nomicous/backend` empty.
- **C1 (partial)** — 5 greps deleted where an executing test already proved the property.

## Open
- **C1 remainder**: `test_deployment_hardening.py:160-164` (Dockerfile greps),
  `test_device_lease_wiring.py:140-171`, `test_device_pairing.py:243-244`.
- **C2**: all 8 executable replacements (reset-script guards via stubbed `psql`; role grants via
  `pg_roles`/`role_table_grants`; advisory try-lock across two sessions; `asyncio.to_thread`
  thread-id spy; migration columns via `information_schema`; `build.sh DEST="/"`; font resolver
  equality + Greek `_render_pdf`; pip-uninstalled as a CI step).

## Suite state
- Integration lane baseline **without** this branch's new file: 4 failures, all
  `test_device_pairing.py`, all the known asyncpg "attached to a different loop" issue (#63).
- One full-lane run **with** the new file also showed
  `test_device_lease.py::test_a_platform_dispatched_page_still_fails_on_the_waiting_timeout` red.
  Re-running that file together with the new file passes 17/17, twice. **Unresolved** — treat as a
  suspected ordering/loop interaction, not as cleared. Re-run the full lane before merging.
- Frontend `node_modules` symlink was removed from the worktree; recreate it to run `npm test`:
  `ln -s <repo>/nomicous/frontend/node_modules .claude/worktrees/test-hardening/nomicous/frontend/node_modules`
