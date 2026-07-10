# Repository hygiene

**Date:** 2026-07-10
**Status:** Recommendations for review — describes current habits, known drift, and
suggested improvements.

This repo mixes **research** (Calamari training, notebooks, `data/`) with
**production** (Nomicous, inference, landing). Hygiene is strong in places (CI,
issues tracker, ADRs) but **drift between what shipped and what docs say** is
the main recurring risk.

Related:

| Doc | Topic |
|-----|--------|
| [repository-cleanup-plan.md](repository-cleanup-plan.md) | Dead code audit — files, CSS, deps, suggested PR series |
| [codebase-audit.md](codebase-audit.md) | Full-stack audit — security, performance, launch readiness |
| [frontend/nextjs-migration.md](frontend/nextjs-migration.md) | Completed Next.js migration record |
| [guides/testing.md](guides/testing.md) | Test lanes and CI commands |
| [issues/README.md](../issues/README.md) | File-based backlog and kanban |

---

## What is already in good shape

### Documentation spine

- [`docs/README.md`](README.md) is the canonical index.
- ADRs live under [`docs/adr/`](adr/) (replacing the older `docs/decisions/` layout).
- [`nomicous/CONTEXT.md`](../nomicous/CONTEXT.md) and
  [`inference/CONTEXT.md`](../inference/CONTEXT.md) sit next to the code they
  describe — good for humans and agents.

### Issue tracking

- File-based [`issues/`](../issues/) with `kanban.md`, `dag.md`, `board.json`.
- Completed work archived under `issues/done/` — not lost when backlog shrinks.

### CI and local gates

- **PR CI** (`quality.yml`): frontend typecheck/lint/test/build; Python ruff +
  unit tests; Postgres integration (`-m "not ml"`); OpenAPI contract check.
- **Security** (`security.yml`): gitleaks, `npm audit`, `pip-audit`.
- **Pre-commit**: Ruff + ESLint + Prettier on the paths that matter.

### Environment boundaries

- Active `.env*` files are gitignored; committed `*.example` templates only.
- Scoped env files per service (platform API vs inference worker).

### Monorepo layout

Clear top-level buckets:

```text
nomicous/     # Production app (API + frontend + migrations)
inference/    # Inference service + registry
landing/      # Marketing site
deploy/       # Vercel / Railway / Fly configs
tests/        # Cross-cutting test suites
scripts/      # Platform seeds, HF publish, migrate helpers
src/          # Research / training code (Calamari, Kraken)
docs/         # Guides, deployment, ADRs
issues/       # Backlog and archive
```

Bounded contexts inside `nomicous/backend/` (`users`, `project`, `document`,
`annotation`, `ml`, `jobs`) are clean.

---

## Known hygiene debt

### H1 — Doc reorg in flight / broken index links

The working tree may contain a **large docs migration** not yet on `main`:

| Removed (legacy) | Replaced by |
|------------------|-------------|
| `docs/decisions/` | `docs/adr/` |
| `docs/architecture/` | ADRs + service READMEs |
| `docs/todo.md` | `issues/` backlog |

**Risk:** `docs/README.md` may still link to files not on every branch (e.g.
`codebase-audit.md`) — a broken index is worse than no index. The cleanup
audit lives in [repository-cleanup-plan.md](repository-cleanup-plan.md).

**Fix:** Land the full reorg in **one atomic commit**: delete old paths, add new
ones, verify every `*.md` link resolves. Treat “docs index has zero broken
links” as a merge gate.

---

### H2 — Stale “Vite / React SPA” references

Next.js App Router migration is complete, but several entry points still
describe the old stack:

| File | Stale text |
|------|------------|
| [`README.md`](../README.md) | “React SPA” for `app.nomicous.com` |
| [`nomicous/README.md`](../nomicous/README.md) | “Vite frontend” |
| [`docs/frontend/performance-optimization.md`](frontend/performance-optimization.md) | “Current stack: Vite + React” (header) |
| [`docs/frontend/nextjs-migration.md`](frontend/nextjs-migration.md) | Written as a plan; should read as **completed** |

**Fix:**

1. Mark `nextjs-migration.md` as **Done** with a short “what changed” section.
2. Update `performance-optimization.md` header to “Next.js App Router”.
3. One-line fixes in root `README.md` and `nomicous/README.md`.

---

### H3 — `.gitignore` gaps

Build and test caches are not ignored:

| Path | Risk |
|------|------|
| `nomicous/frontend/.next/` | Easy to `git add .` by mistake (~11 MB locally) |
| `.pytest_cache/` | Same |

**Suggested additions:**

```gitignore
# Next.js / frontend build output
.next/
.turbo/
*.tsbuildinfo

# Pytest
.pytest_cache/
```

---

### H4 — Stray root `package-lock.json`

Root [`package-lock.json`](../package-lock.json) is nearly empty (`"packages":
{}`). The real lockfile is
[`nomicous/frontend/package-lock.json`](../nomicous/frontend/package-lock.json).

**Fix:** Delete the root lockfile unless introducing npm workspaces at the repo
root.

---

### H5 — Model weights tracked in git

`src/hf/cache/` and `src/hf/local/` contain `.pt` fixture weights (~6 MB each,
multiple files tracked). Pack size is manageable today but the pattern does not
scale.

**Fix:** Document an explicit policy (see [Weight artifact policy](#weight-artifact-policy)
below). Prefer Hub + download script over growing git blobs.

---

### H6 — Pre-commit does not mirror CI

Pre-commit runs **lint and format only**. CI also runs **tests**, **build**, and
**`check:api`**.

**Fix:** Add a documented local check script (see [Local check script](#local-check-script))
and a short `CONTRIBUTING.md`. Do not put Postgres integration in pre-commit
(too slow).

---

### H7 — `issues/README.md` drift

The backlog table may reference issue files that moved to `issues/done/` (e.g.
`034` listed under `backlog/` but archived under `done/huggingface/`).

**Fix:** When closing an issue, update in the **same commit**:

- `issues/README.md`
- `issues/kanban.md`
- `issues/board.json`

---

### H8 — Root README serves two audiences

[`README.md`](../README.md) (~300 lines) covers Calamari CER results, training
layout, Nomicous setup, and deployment. Newcomers cannot quickly tell whether
they need the **research** or **product** path.

**Fix:** Keep one README but add a clear fork at the top:

```markdown
## I want to…

- **Run the annotation app** → [nomicous/README.md](nomicous/README.md)
- **Train / evaluate OCR models** → [Training](#training-src-and-inference-inference) (below)
- **Deploy to production** → [docs/deployment/production.md](deployment/production.md)
```

Optional later: move training content to `docs/research.md` and shorten the
root.

---

### H9 — Dead-code audit trail

A dead-code pass removed modules (`ImageCanvas/`, pairing strip, etc.). Ongoing
candidates are recorded in [repository-cleanup-plan.md](repository-cleanup-plan.md).

**Fix:** Follow the [Dead-code ritual](#dead-code-ritual) below; update the plan
before each cleanup PR.

---

### H10 — Git worktree hygiene

With parallel worktrees (e.g. `main` + `nextjs-opt`):

| Practice | Why |
|----------|-----|
| One concern per worktree/branch | Avoids mixed doc reorg + perf + cleanup in one diff |
| Commit shared docs to `main` first | Worktrees inherit the index |
| Document worktree paths | See [guides/local-development.md](guides/local-development.md) |
| Never commit `.next/` | Fix `.gitignore` before branching |

Example:

```bash
git worktree add ../nextjs-opt -b nextjs-opt
git worktree list
git worktree remove ../nextjs-opt   # when done
```

---

## Suggested improvement roadmap

### Quick wins (< 1 hour)

| # | Action |
|---|--------|
| 1 | Add `.next/`, `.pytest_cache/` to `.gitignore` |
| 2 | Delete stray root `package-lock.json` |
| 3 | Fix Vite → Next.js one-liners in `README.md`, `nomicous/README.md` |
| 4 | Restore or remove broken links in `docs/README.md` |
| 5 | Fix `issues/README.md` backlog table |

### Medium effort (half day)

| # | Action |
|---|--------|
| 6 | Land doc reorg atomically (`decisions/` → `adr/`, etc.) |
| 7 | Mark `nextjs-migration.md` complete; refresh performance doc header |
| 8 | Add `scripts/check.sh` + `CONTRIBUTING.md` |
| 9 | Add `src/hf/README.md` with weight artifact policy |

### Recommended first hygiene PR

Single PR, e.g. `docs: land ADR reorg and repo hygiene fixes`:

1. Commit full `docs/` reorg (adr, frontend, deployment hardening docs).
2. Keep [repository-cleanup-plan.md](repository-cleanup-plan.md) in sync after each cleanup PR.
3. `.gitignore` + delete root `package-lock.json`.
4. Update Vite → Next.js in READMEs.
5. Fix `issues/README.md` backlog entry.

---

## Policies and rituals

### Dead-code ritual

1. **Before deleting:** grep for imports; record candidates in
   `docs/repository-cleanup-plan.md` with confidence (high / medium / low).
2. **After deleting:** run `npm test`, `npm run build`, targeted pytest.
3. **On merge:** commit message references the plan, e.g.
   `chore: remove dead ImageCanvas tree (see repository-cleanup-plan.md)`.

High-confidence removal method (from prior audit):

1. Trace from entry points (`create_app()`, Next.js `src/app/`).
2. Grep importers of each candidate module.
3. Classify: **high** (zero external importers), **medium** (tests/docs only),
   **low** (needs product decision).
4. Remove high-confidence code; trim dead exports in live files.

---

### Weight artifact policy

Pick one approach and document it in `src/hf/README.md`:

| Approach | When to use |
|----------|-------------|
| **Git LFS** for small fixture weights used in CI | Tests require real weights locally and in CI |
| **Hub-only** + `.gitkeep` placeholders | Production models (preferred) |
| **Download script** (`scripts/hf/pull-fixtures.sh`) | Dev/CI pulls on demand |

Current state: small `.pt` fixtures are tracked under `src/hf/cache/` and
`src/hf/local/`. Revisit before adding larger checkpoints.

---

### Local check script

Add `scripts/check.sh` as a documented “pre-PR” mirror of CI (fast lane only):

```bash
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

echo "== Frontend =="
npm --prefix "$root/nomicous/frontend" run typecheck
npm --prefix "$root/nomicous/frontend" run lint
npm --prefix "$root/nomicous/frontend" test
npm --prefix "$root/nomicous/frontend" run build

echo "== Python unit =="
cd "$root"
uv run --locked --group platform --group inference \
  pytest tests/nomicous/unit tests/inference/unit tests/hf -q

echo "OK"
```

Full integration (Postgres, ML) stays in CI — see [guides/testing.md](guides/testing.md).

---

### Docs link check (manual)

Before a release, spot-check that index links resolve:

```bash
# From repo root — list markdown links in docs/README.md and verify targets exist
rg -o '\[[^\]]+\]\(([^)]+\.md[^)]*)\)' docs/README.md
```

Automating this in CI is optional; a monthly manual pass is enough at current
scale.

---

### Ongoing habits

| Ritual | Frequency |
|--------|-----------|
| Dead-code pass after large refactors | Per feature |
| Docs index link check | Before release |
| `issues/kanban.md` matches filesystem | When closing issues |
| Review `pip-audit` / `npm audit` ignores | Monthly |
| Archive completed migration specs | When milestone ships |

---

## Research vs product boundary

The repo does not need to split into two repositories yet. Document the mental
model explicitly:

```text
Research                         Product
────────                         ───────
src/                             nomicous/
experiments/                     inference/
configs/                         landing/
data/ (local, gitignored)        deploy/
outputs/ (gitignored)
```

**Rule of thumb:** changes under `nomicous/`, `inference/`, `landing/`, and
`deploy/` follow production CI and deployment docs. Changes under `src/` and
`experiments/` follow training notebooks and config conventions. Cross-cutting
changes (registry, HF publish) touch both — note that in PR descriptions.

Optional future rename (not required now):

```text
research/   ← src/, experiments/, configs/
product/    ← nomicous/, inference/, landing/
```

---

## Pre-commit vs CI matrix

| Check | Pre-commit | PR CI |
|-------|------------|-------|
| Ruff check / format | Yes | Yes |
| ESLint / Prettier | Yes | Yes |
| Frontend typecheck | No | Yes |
| Frontend tests | No | Yes |
| Frontend build | No | Yes |
| Python unit tests | No | Yes |
| Postgres integration | No | Yes |
| OpenAPI drift (`check:api`) | No | Yes |
| gitleaks / dependency audit | No | Yes |
| Docker / deployment build | No | Disabled (`deployment.yml`) |

Gap to close: re-enable deployment workflow or equivalent image-build gate —
see [codebase-audit.md](codebase-audit.md) if present.

---

## Review checklist

Use when triaging hygiene work:

- [ ] `docs/README.md` has no broken links
- [ ] `.gitignore` covers `.next/` and `.pytest_cache/`
- [ ] No stray root `package-lock.json`
- [ ] READMEs say Next.js, not Vite/React SPA
- [ ] `issues/README.md` matches `backlog/` and `done/`
- [ ] Dead-code removals recorded in `repository-cleanup-plan.md`
- [ ] `src/hf/` weight policy documented
- [ ] `scripts/check.sh` exists and is mentioned in `CONTRIBUTING.md`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-10 | Initial hygiene assessment and recommendations documented |
