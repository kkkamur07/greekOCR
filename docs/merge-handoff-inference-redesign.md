# Merge handoff — inference redesign session (2026-08-04, later)

**Read this alongside [`merge-handoff.md`](./merge-handoff.md), which it supersedes in part.**
That document's branch table is now wrong in ways that matter: it directs a merge of
`feat/048-collapse-second-job-queue`, a branch that **no longer exists**. See §2.

Trunk is **`feat/inference-cli-redesign`**, now at the `origin/main` merge (`2a70f8e`),
not `a0bc170`.

---

## 1. The one thing a merge must not silently undo

**ADR 0004 reversed the inference runtime: PyTorch replaces ONNX Runtime.**

This is the highest-risk item in the whole session, because the reversal is *invisible in a
diff*. Two live branches — `feat/deep-cleanup` and `feat/frontend-libraries` — contain
edits to files that issue 049 **archives and deletes**:

```
inference/architectures/calamari/onnx.py
inference/architectures/calamari/adapter.py
inference/architectures/blla/onnx.py
inference/architectures/blla/blla_runtime.py
inference/architectures/blla/blla_preprocessing.py
inference/architectures/blla/blla_decoder/__init__.py
```

A plausible three-way merge resolves "modified on one side, deleted on the other" by
**keeping the modification**. If that happens, the ONNX runtime comes back, the parity
problem comes back with it, and nothing fails loudly — the code still works, it is simply
the architecture we decided against.

**Rule: on any modify/delete conflict under `inference/architectures/`, the deletion
wins.** The ONNX implementation lives at `archive/onnx-runtime/` by design, not by
accident. If a merge produces a tree where both `archive/onnx-runtime/` and a live
`architectures/**/onnx.py` exist, the merge is wrong.

Verification after merging: `onnxruntime` must not appear in the runtime dependency
closure, and `grep -rn "onnxruntime" inference/` must return nothing outside `archive/`.

## 2. What happened to `feat/048-collapse-second-job-queue`

**Deleted. Do not attempt to merge it.** The older handoff calls it "the only real merge";
that is no longer true.

It had branched from `21c24b2` — before the device-pairing commit and before this session
— so it carried neither ADR, nor the issue board, nor `architectures/{artifact,isolation}.py`.
The agent working it was told to read `docs/adr/0003-*.md` and its issue spec, and **neither
file existed in its worktree**; it worked from its prompt text alone.

Its two substantive commits (`c2a5ff0`, `99ca23f` — roughly 2,400 deletions, the correct
queue collapse) were cherry-picked onto the corrected base and produced **6 conflicts**,
all in deletion-heavy hunks where a bad resolution leaves a half-deleted tree that tests do
not reliably catch. The cherry-pick was aborted and the work relaunched cleanly on
`feat/048-queue-collapse`.

The two commits below those two (`516c3fc`, `4b1262f`) were never that agent's work — they
are a collaborator's, already on `origin/main`, and arrived via the wrong base. They are now
on trunk through an ordinary merge.

Also deleted: `feat/049-torch-free-runtime-boundary`. Its issue was **inverted** by ADR
0004 — it originally required making the runtime Torch-*free*. Merging anything from it
would fight ADR 0004 directly.

## 3. Branch topology now

| Branch | Status |
|---|---|
| `feat/inference-cli-redesign` | **trunk** — merged `origin/main` (astha's 2 commits), zero conflicts |
| `feat/048-queue-collapse` | in flight — queue collapse, ADR 0003 |
| `feat/049-torch-runtime-archive-onnx` | in flight — Torch runtime, ADR 0004 |
| `feat/deep-cleanup` | live worktree, 234 files — **overlaps §1** |
| `feat/frontend-libraries` | live worktree, 264 files — **overlaps §1** |
| `feat/048-collapse-second-job-queue` | **deleted** — see §2 |
| `feat/049-torch-free-runtime-boundary` | **deleted** — inverted by ADR 0004 |

`origin/main` moved during the session (a collaborator is pushing to it). Trunk has been
merged with it once; check again before merging.

## 4. Other invariants that a merge can quietly revert

**One job queue (ADR 0003).** Issue 048 deletes `inference/infrastructure/` entirely — the
inference service's own database, ORM, repository, and settings. A merge that restores any
of it recreates the second queue. `psycopg2` and `sqlalchemy` must not return to the
`inference` dependency group, and the `inference-api` container must stay gone.

**Route tests must go through the real `create_app()` (ADR 0001).** An earlier device
layer was never mounted, and the integration suite hid it by constructing its own FastAPI
app in the test file — the entire phase was unreachable behind a green suite. If a merge
reintroduces a test-local FastAPI app, it restores the blindness, not just a test style.

**Tests are live in this work.** No mocks, no fake transports, no in-memory database
substitutes; where a live test was not achievable the test was deferred and recorded rather
than replaced with a mock. A merge that resurrects a mocked version of a test deleted here
is a regression even though the suite goes green.

**`excludes.txt` and the bundle verifier are gone by construction, not by boundary.** ADR
0002 originally planned to delete them via a package boundary that kept Torch out. ADR 0004
made Torch the runtime, so there is no forbidden dependency left to police. If either file
reappears, something merged ADR 0002's superseded packaging section.

## 5. Decisions recorded this session

- `docs/adr/0004-pytorch-is-the-inference-runtime.md` — **new.** PyTorch is the runtime;
  ONNX archived under `archive/onnx-runtime/` with commentary, at the owner's request,
  because the conversion work was substantial.
- `docs/adr/0002-…` — packaging section marked **superseded in part** by 0004.
- `inference/CONTEXT.md` — **Hub artifact** redefined to native PyTorch checkpoints; two
  flagged-ambiguity entries updated. Expect conflicts here; the ADR 0004 wording wins.
- GitHub: PRD [#47](https://github.com/kkkamur07/greekOCR/issues/47), issues #48–#61.
  #49 and #50 were inverted by ADR 0004 and carry comments explaining why.

## 6. Known-unresolved

**PyTorch CPU latency is unmeasured.** ADR 0004 accepts the risk explicitly rather than
claiming it was assessed — a benchmark was commissioned and stopped before producing
numbers. ONNX Runtime generally outperforms Torch eager on the small convolutional and
recurrent graphs these models use, and for local inference that latency *is* the product
experience. Issue #49 requires before/after single-page latency be recorded. **If that
number is bad, ADR 0004 says to revisit the decision, not to absorb it.** Do not let this
fall off the list during a merge.

`docs/merge-handoff.md` §1's branch table and its claim that
`feat/048-collapse-second-job-queue` is the only real merge are void. The rest of that
document — particularly the `001_initial_schema.py` / `create_all()` migration invariant —
still stands and is unaffected by anything here.
