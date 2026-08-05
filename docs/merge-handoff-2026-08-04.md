# Merge handoff, 2026-08-04

Two branches are finished and waiting to merge, each living in its own worktree
on disk. This note says what is in them, what order to take them in, and what
not to re-derive. It is disposable: delete it once both are merged.

Trunk for this work is `feat/inference-cli-redesign`. At the time of writing it
is 6 commits ahead of `origin/main` and 0 behind, having just absorbed
`origin/main` at `4b1262f`.

## What is waiting

| Branch | Tip | Worktree | Ahead of trunk | Behind trunk |
| ------ | --- | -------- | -------------- | ------------ |
| `feat/deep-cleanup` | `6f49a4e` | `../greekOCR-deepclean` | 1 | 7 |
| `feat/frontend-libraries` | `c4752a6` | `../greekOCR-frontend` | 1 | 5 |

Each branch is exactly one commit. Neither needs a merge commit or a rebase of a
series: cherry-pick the single commit onto the trunk, or rebase the branch, and
delete it.

Both commit messages carry the full reasoning, including the changes that were
deliberately *not* made and why. Read them before reviewing the diffs.

### `feat/deep-cleanup` (`6f49a4e`)

33 files, +929/-3531. Dead code removal, generated API types replacing
hand-written ones in `client.ts`, a `segmentNumbering` module that gives the
Segment number rule one home instead of three disagreeing copies, mypy added to
the `dev` group and to CI on `continue-on-error`, and a corrected skip guard in
`test_blla.py`.

Verified in its worktree: `tsc` clean, 48 test files and 197 tests passing, ruff
clean, 680 Python tests collected with no errors.

### `feat/frontend-libraries` (`c4752a6`)

16 files, +246/-409. Replaces the hand-rolled server-state cache with TanStack
Query and the hand-rolled toast stack with antd's `message`, which was already a
direct dependency and imported nowhere.

Verified in its worktree: `tsc` clean, 47 test files and 187 tests passing,
eslint 0 errors, `next build` succeeds.

## Suggested order

Take `feat/deep-cleanup` first, then `feat/frontend-libraries`.

The reason is not dependency, it is base age. Deep-cleanup branched at `0b998ec`
and its frontend edits were written against the frontend as it was *before* the
server-state query layer existed. Frontend-libraries branched at `af4ac95`,
after it. Landing the older base first means the newer one rebases onto a tree
it already understands.

## Where they touch the same files

Almost nowhere, which is the good news.

Deep-cleanup edits `client.ts`, `PageEditorCanvas.tsx`, `useLayoutMutations.ts`,
`usePairingState.ts`, `characterConfidence.ts`, `hooks/utils.ts`, `types.ts`,
`index.css`, `page-editor.css`. Frontend-libraries edits `queryClient.ts`,
`useServerQuery.ts`, `resources.ts`, `providers.tsx`, `toast.ts`, `storage.ts`,
`usePageEditorData.ts`, `helperInfo.ts`, `theme-shell.css`, `vitest.setup.ts`.
Disjoint sets.

The one real collision is `nomicous/frontend/package.json`. Deep-cleanup removes
four unused dependencies, frontend-libraries adds `@tanstack/react-query`. Both
edits belong in the result. Both branches also carry a regenerated
`package-lock.json`, so that file will conflict too: take either side, then run
a single `npm install` to reconcile it rather than hand-resolving 3,000 lines.

Re-run the frontend suite after the second merge. Neither branch has been tested
against the other.

## Constraints that apply to this work

**Never edit anything under `src/`.** Auditing and reporting on it is wanted;
editing it is not. This is a standing instruction from the repo owner, given
2026-08-04. Several otherwise-actionable findings die here, listed below.

`nomicous/backend/` was hands-off for the frontend task specifically. Whether
that still holds more generally is unresolved. Ask before editing it.

## What not to re-derive

**ADR 0004 supersedes most of the inference audit.** It is Accepted but not yet
implemented: `archive/onnx-runtime/` does not exist and every ONNX file is still
in place. Findings about the manual sigmoid in `blla_decoder/__init__.py`, the
shoelace area in `segment_geometry.py`, and the Moore-neighbour contour tracer
in `blla_decoder/lines.py` all sit in the path being archived. Do not spend
effort on them. The installer-size reasoning that justified several "keep"
verdicts in that audit died with ADR 0002.

**`docs/codebase-review-2026-08-04.md` is stale.** It predates ADR 0004 and the
`src/` constraint. Treat it as a record of what was believed that day, not as a
work list.

**`useInferenceHost` keeps its hand-written focus listeners on purpose.**
TanStack Query v4 dropped the window `focus` listener and its focus manager
listens only to `visibilitychange`, which does not fire when a user alt-tabs
back from another application. That is the exact case the hook exists for.
`getCache.ts` also stays on purpose: it dedupes in-flight GETs beneath the
transport, serving callers that never go through a hook.

## Findings raised but not acted on

Inside `src/`, therefore report-only:

- `src/preprocessing_data/syriac/xml_to_data.py:288` splits Syriac train, val
  and test as contiguous alphabetical slices with no shuffle. Adjacent
  manuscript pages share scribe, ink and layout, so the published Syriac CER is
  optimistic by an unknown margin. The owner will fix this and re-measure.
- `src/model/calamari/` is a 109-file vendored TensorFlow Calamari. Still the
  only in-repo Calamari trainer, reached by subprocess from
  `src/train/calamari/train_utils.py:40`. `calamari_ocr/utils/grayscale.py`
  (34 lines) is locally authored and load-bearing for the parity test.
- `src/preprocessing_data/estebanData.py` is orphaned and imports pandas, which
  is declared nowhere.
- `_resolve_path` is duplicated verbatim in two files and the image-extension
  tuple in four.

Outside `src/`, still open:

- `pyproject.toml` declares `jiwer` and `accelerate` in the `train` extra with
  zero import sites anywhere.
- hydra, omegaconf, lightning, wandb and pandas are imported by training code
  and declared in no manifest. A fourth environment is required and only a
  source comment says so.
- The 43.6 MB `trOCR.ipynb` blob is still in git history. Reclaiming it needs
  `git filter-repo`. Not decided.
- The staged `blla.onnx` digest differs from both the `registry.yaml` pin and
  the published Hub artifact. Anyone resolving from the Hub today gets the older
  graph. Now entangled with ADR 0004.

## Worktrees on disk

```
../greekOCR              feat/inference-cli-redesign   trunk, clean
../greekOCR-deepclean    feat/deep-cleanup             clean, own node_modules
../greekOCR-frontend     feat/frontend-libraries       clean, own node_modules
```

Both extra worktrees have their own real `node_modules`, not symlinks. An
earlier symlink into the main tree's `node_modules` was removed because
installing through it would have pruned the dependencies deep-cleanup deletes
and corrupted the main tree. Do not re-create it. Neither worktree has its own
`.venv`; both borrow the main tree's for read-only checks such as ruff and
collect-only. Do not run an install or sync against it from a worktree, since
that prunes packages the main tree needs.

Remove both worktrees with `git worktree remove` once merged.
