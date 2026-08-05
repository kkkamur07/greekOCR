# 0004. PyTorch is the inference runtime; ONNX is archived

- **Status:** Accepted
- **Date:** 2026-08-04
- **Builds on:** [0002](./0002-inference-cli-replaces-loopback-helper.md), which moved
  distribution from frozen native installers to `uv tool install` from PyPI.

## Context

ONNX Runtime was chosen when local inference shipped as a frozen native
application. In that world two properties were load-bearing:

- Every megabyte shipped through a four-platform build, Developer ID
  notarization, and Authenticode signing.
- There was no update channel, so the vendored dependency tree we distributed
  was the one researchers kept running.

Both properties made a small, dependency-light runtime worth real complexity: an
export pipeline, a second set of published artifacts, a `Torch`-exclusion
denylist (`excludes.txt`), a bundle verifier, and a parity apparatus to prove the
converted graph still matched the trained one.

ADR 0002 deleted the frozen bundle. Its own packaging section, however, carried
the old constraint forward unexamined — it required Torch to be moved *out* of
the published package and treated the denylist's removal as the win. That was
the bundle-era premise surviving its own justification.

## Decision

**PyTorch is the inference runtime. ONNX Runtime, its conversion scripts, and the
ONNX↔Torch parity tooling are archived under `archive/onnx-runtime/`.**

Archived rather than deleted: git history is a poor place to find a retired
subsystem, and the archive carries a README naming what it was and why it was
retired.

## Rationale

**The constraint that paid for ONNX is gone.** `uv tool install` has neither
property that made a lean runtime worth an export pipeline: there is no signing
cost per megabyte, and there *is* an update channel — enforced by the version
floor in ADR 0002. The reasoning that justified ONNX did not survive the change
in distribution, and keeping the conclusion after deleting its premise is how
architectures accumulate cost nobody can account for.

**Training is already PyTorch.** One framework end to end removes the conversion
step, and with it the entire class of defect where what runs has drifted from
what was trained. That is what the parity apparatus existed to detect; with one
runtime there is nothing to be at parity *with*.

**The Torch graphs already exist and already run.** This is a deletion, not a
port:

| | lines |
|---|---|
| `src/model/inference_export/calamari/{model,layers,config}.py` | 279 |
| `inference/architectures/blla/blla_model.py` | 142 |
| `inference/architectures/calamari/{model,layers,config}.py` (re-export shims) | 41 |

`blla.py` already loads a state dict and calls `.eval()`. The work is removing
the ONNX loaders and pointing the runtime at graphs that are present today.

**Native checkpoints are already published.** `best.pt` and `blla.safetensors`
sit beside the `.onnx` artifacts in the staging tree, and the offline bundled
weights are `.pt` only. No re-publishing, no new **artifact SHA-256** story — the
digests move to the artifacts already there.

**Several things stop having a reason to exist**: `excludes.txt`, the bundle
verifier, the `kraken` parity dependency, and dual-format artifact publishing.
ADR 0002 wanted the first two dead via a package boundary; this kills them by
construction instead, because there is no forbidden dependency left to police.

## Costs accepted

**Install size.** Measured on the development machine: `onnxruntime` is 63 MB
installed against `torch` at 388 MB. Accepted as a one-time cost — `uv` caches,
and a version bump of the published package does not re-fetch Torch unless the
Torch pin itself moves. The package must pin the CPU-only build; CUDA wheels
dominate the download and are useless on a researcher's laptop.

**CPU latency — accepted without measurement.** ONNX Runtime generally
outperforms PyTorch eager on the small convolutional and recurrent graphs these
models use, and for local inference that latency *is* the product experience. A
benchmark was commissioned and stopped before producing numbers, so **this risk
is accepted unmeasured rather than assessed**. It is not thereby resolved: the
implementing issue requires single-page CPU latency to be recorded before and
after the switch, so the real cost lands on the record during the work. If it
proves severe, this record should be revisited rather than quietly tolerated —
the mitigation is `torch.compile` or a return to a compiled runtime for the
transcription path only, not silent acceptance.

**Checkpoint loading becomes a code-execution surface.** Loading a pickled Torch
checkpoint executes code in a way loading an ONNX graph does not. Mitigated by
preferring `safetensors`, keeping **artifact SHA-256** verification ahead of the
load, and never unpickling an unverified file — not eliminated.

## Consequences

- The published package carries a large dependency. Cold-install time on a poor
  connection becomes a real part of the first-run experience, and should be
  measured rather than assumed.
- The export tree stops being a build stage and becomes archived reference.
- ADR 0002's "One published package, not two" section is amended: the Torch
  modules stay inside the published package rather than moving out, and the
  denylist dies by construction rather than by boundary.
- Security patching keeps the shape ADR 0002 gave it — a dependency bump plus a
  version-floor bump — but the surface is now larger, since Torch and its
  transitive tree are in it.

## Alternatives considered

**Keep ONNX Runtime.** Rejected: its justification was the frozen bundle, which
no longer exists. Retaining it means maintaining an export pipeline, a second
artifact format, a denylist, and a parity harness to serve a size constraint
nothing now imposes.

**Ship both runtimes and select at load.** Rejected — this is precisely the
parity burden, made permanent and given a configuration surface. Two runtimes for
one set of models is the cost being removed, not a compromise between the options.

**Delete the ONNX code outright rather than archiving it.** Rejected on the
owner's instruction, and reasonably: the conversion work is non-trivial and
rediscovering it from history is far harder than reading it from `archive/`.
