# Repository cleanup plan

**Date:** 2026-07-10  
**Status:** Audit for review — candidates only; nothing removed by this document.

Use with the [dead-code ritual](repository-hygiene.md#dead-code-ritual) before deleting
code. Launch blockers live in [codebase-audit.md](codebase-audit.md).

---

## Executive summary

| Area | Finding |
|------|---------|
| `nomicous/frontend/` | Vite-era leftovers, stale CSS, unused npm deps — no orphan components |
| `nomicous/backend/` | Dead symbols and env templates — all modules wired |
| `inference/` | Repeated constants; Greek model unpinned (see audit) |
| `src/preprocessing_data/` | Legacy scripts — no inbound imports |
| Repo-wide | Doc index fixed on `main`; issue tracker drift on some branches |

**Prior removals:** `ImageCanvas/`, pairing strip, `ModelSettings`, trimmed exports.
See [ADR 0003](adr/0003-image-canvas-archival-boundary.md).

---

## Quick wins (high confidence)

### Frontend

| # | Item | Path |
|---|------|------|
| F1 | Vite SPA shell | `nomicous/frontend/index.html` |
| F2 | Dead navigation helper + test | `src/pages/pageEditorNavigation.ts` |
| F3 | Unused npm deps | `eslint-plugin-react-refresh`, `eslint-config-next`, `@ant-design/icons`, `playwright` |
| F4 | Stale CSS | `src/styles/page-editor.css`, `theme-shell.css`, `index.css` |

### Backend

| # | Item | Path |
|---|------|------|
| B1 | Unused exception | `DatabaseUnavailableError` in `core/exceptions.py` |
| B2 | Dead function | `process()` in `annotation/application/processing.py` |
| B3 | Stale env template | `nomicous/backend/.env.example` → pointer to `core/.env.example` |
| B4 | Duplicate env line | `INFERENCE_URL` twice in `core/.env.example` |

### Repo-wide

| # | Item | Path |
|---|------|------|
| R1 | Empty root lockfile | `package-lock.json` (repo root) |
| R2 | Orphan script | `scripts/verify-hardening.sh` |
| R3 | Notebook merge conflict | `src/experiments/calamari.ipyn_greek.ipynb` |

---

## Medium confidence

- Consolidate `CharacterConfidence` types onto OpenAPI (`api/client.ts`)
- Stop exporting `confidenceHighlightColor`, `approvedText` (tests-only or internal)
- Duplicate `_geometry_points()` in backend — merge into `line_geometry`
- Duplicate settings validators in platform vs inference
- Archive `src/preprocessing_data/` after team confirm
- Rename `PageEditorPlaceholderPage` → `PageEditorPage` (naming debt)

---

## Suggested PR series

1. **Hygiene** — F1–F3, R1, B3–B4 (~1 h)
2. **CSS prune** — F4 (~1–2 h)
3. **Backend symbols** — B1–B2 (~1 h)
4. **Type consolidation** — medium frontend items (~½ day)
5. **Docs pass** — Vite→Next drift in READMEs (~½ day)

---

## Review checklist

- [ ] Candidate recorded here with confidence level
- [ ] `grep` shows no remaining importers
- [ ] `npm test` + `npm run build` (frontend)
- [ ] `pytest tests/nomicous/unit` (backend)
- [ ] Commit references this plan

Full detail: expand sections in this file as cleanups land.
