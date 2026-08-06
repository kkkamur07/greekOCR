# 0006. ONNX Runtime is the inference runtime; PyTorch builds the artifact

- Status: Accepted
- Date: 2026-08-05
- Supersedes: [0004](./0004-pytorch-is-the-inference-runtime.md), which made PyTorch the runtime
  and archived ONNX Runtime.
- Builds on: [0002](./0002-inference-cli-replaces-loopback-helper.md), which moved distribution
  from frozen native installers to `uv tool install` from PyPI.

## Context

ADR 0004 reasoned that the constraint which had paid for ONNX Runtime — a frozen four-platform
bundle where every megabyte went through notarization and there was no update channel — was
gone, so the conversion step was no longer worth its cost. That reasoning was sound about the
*old* constraint. It did not put a number on the new one until after the decision was taken,
and its own "Costs accepted" section is where the case falls apart:

- **817 MB installed on macOS arm64**, measured again on 2026-08-05 against the same published
  dependency set. 475 MB of that is Torch, plus SymPy, networkx and torchgen behind it.
- **Roughly 980 MB fetched on Linux with the CPU pin**, and **4801 MB without it** — sixteen
  `nvidia-*` and `triton` wheels that no researcher's laptop can use.
- The CPU pin **cannot be expressed in package metadata**. No PEP 621 field names an index and
  `tool.uv.sources` is ignored when a package is consumed from a registry, so ADR 0004 pushed
  the pin onto the user as `uv tool install nomicous-inference --torch-backend=cpu` — and
  recorded two residual risks it could not solve: plain `pip` gets the 4.8 GB tree, and uv
  0.7.x reads `UV_TORCH_BACKEND=cpu` and then silently installs the CUDA tree anyway.
- **Peak RSS on a page more than doubled**, 3.0 GB to 7.0 GB, which ADR 0004 called "the
  sharpest edge of this decision" on an 8 GB laptop.
- **Intel macOS dropped off the supported set** entirely, because PyTorch publishes no
  `x86_64-apple-darwin` wheel from 2.10 onward.

What was bought for that: a 12–14% median latency win on the whole page, from a forward pass
that is 40% faster but is only about a fifth of segment latency — the NumPy and scikit-image
decoder dominates at ~5.0 s either way — and the removal of a conversion step.

That is the trade this record reverses. A worse first-run experience, a worse memory ceiling, a
narrower platform matrix and an install command with two ways to silently go wrong, in exchange
for a latency difference the user cannot perceive against a decoder that did not change.

## Decision

ONNX Runtime is the inference runtime for both architectures. PyTorch's role is to *build* the
artifact: the graph definitions and the exporters live in `src/model/inference_export/`, which
is not in the published wheel, and Torch is a `export` dependency group rather than a
`[project]` dependency.

`archive/onnx-runtime/` is deleted. Its code is live again — that directory was the source this
change was restored from, near enough verbatim — and the knowledge it carried that is not
expressed in code is folded into this record.

## What this restores, and what it does not

The adapters came back as they were, with two changes each. They are no longer a *second*
runtime beside a Torch one, so they keep the plain names; and they gained the two things the
live tree grew while they were archived — `resolve_artifact`, so the **artifact SHA-256** is
verified before the file is opened, and `reraise_if_none_survived`, so an all-failed page
re-raises its first cause instead of returning an empty page.

| restored to | from |
|---|---|
| `inference/architectures/calamari/adapter.py` | the archived `calamari/onnx.py` |
| `inference/architectures/blla/blla.py` | the archived `blla/onnx.py` |
| `inference/architectures/blla/blla_preprocessing.py` | `preprocess_blla_image_numpy` |
| `blla_decoder/common.py::resize_heatmaps_nearest` | the same helper, `numpy_support.py` |
| `src/model/inference_export/{calamari,blla}/export.py` | the archived exporters |

`layers.pad_same` regained its tracer branch. Under tracing `x.shape` freezes into Python
constants, so without `torch._shape_as_tensor` the exported graph pads every line to the width
of the single dummy input it was traced on. ADR 0004 dropped that branch as dead code when the
file became runtime-only; it is load-bearing again.

The Kraken-oracle parity harness (`test_blla_parity.py`, 460 lines) was **not** restored. Its
oracle is `kraken>=7.0.2`, which is not in any dependency group, so restoring it would have
added a permanently-skipped file. The claim it made is now made directly against the Torch
graph, which is in this repository, by `tests/export/`.

## The accumulator bug, and what shipping without its fix cost

This is the knowledge the archive existed to preserve, and it turned out to be more than
historical.

`nn.GroupNorm` lowers to `Reshape([0, 32, -1]) -> InstanceNormalization`, which flattens a group
into a single axis of `C/G * H * W` elements. onnxruntime's CPU kernel then accumulates that
group's mean and variance in one serial float32 accumulator — on a 1800×2471 page, 2,224,800
values through one accumulator. On real, spatially correlated post-ReLU activations the rounding
bias accumulates instead of cancelling. `torch.nn.functional.group_norm` uses a blocked/Welford
accumulation and stays in float32 noise, which is why only the ONNX side moves.

The fix is `_ExportGroupNorm`: reduce over the width axis first, then over the remaining group
axis. Identical arithmetic — every block has exactly `width` elements — but no accumulator sees
more than a few thousand terms. It is applied **only while tracing**, because the staged
reduction perturbs the native float32 logits by up to 1.8e-03, which is harmless numerically but
enough to break the Torch graph's bit-exact agreement with the Kraken oracle.

**The `blla.onnx` published at revision `444d51dd` did not have it.** Its graph carries six
`InstanceNormalization` nodes and no `ReduceMean`. So while ADR 0004 was retiring the ONNX
runtime partly on the strength of a parity measurement, the artifact users would actually have
fetched was the unfixed export — and the discrepancy is visible in ADR 0004's own numbers, which
record a BLLA max logit delta of 9.2e-03 that no published artifact could reproduce.

Measured here on `segment_page.jpeg` against the Torch graph:

| `blla.onnx` | logits mean \|Δ\| | p99 | max | baselines identical | min polygon IoU |
|---|---|---|---|---|---|
| `444d51dd` (published, unfixed) | 1.26e-03 | 8.67e-03 | 1.52e-01 | 33/34 | 0.7266 |
| `5c20a584` (re-exported) | 1.05e-05 | 4.48e-05 | 1.55e-03 | **34/34** | **1.0000** |

The drift lands exactly where the mechanism predicts: the two shortest lines on the page, 19 px
and 25 px against a 390 px median, whose polygons restructure when a handful of pixels cross the
0.5 sigmoid boundary. Transcription was never affected — Calamari decodes byte-identical text on
both runtimes, with a confidence delta of 1.1e-07.

The artifact was re-exported and re-published on 2026-08-05, and the registry now pins revision
`5c20a584b39988a25dfc682f9fe634ac1b4a42dd`. `tests/export/test_blla_export_group_norm.py` checks
the **published** file rather than a fresh export, because "the exporter is correct" and "the
artifact is correct" are different claims and it was the second one that was false.

Calamari needed no republish: re-exporting `best.pt` today produces a file byte-identical to the
published `best.onnx`, digest `3cb01b58…`, so its pin moved from the checkpoint to the graph at
the same revision.

## Measured on the switch back

Installed closure of the published wheel and its declared dependencies, into an empty
virtual environment on the development machine (macOS arm64, Python 3.11), `uv pip install`
with a cold cache, measured 2026-08-05 — the same method ADR 0004 used for its 811 MB figure.

|                          | ADR 0004 (Torch) | ADR 0006 (ONNX Runtime) | change |
| ------------------------ | ---------------- | ----------------------- | ------ |
| **Installed closure**    | **817 MB**       | **372 MB**              | **−445 MB, −54%** |
| Runtime package itself   | `torch` 475 MB   | `onnxruntime` 70 MB     | −405 MB |
| Built wheel              | 112 KB           | 108 KB                  | — |
| Install command          | `uv tool install nomicous-inference --torch-backend=cpu` | `uv tool install nomicous-inference` | flag gone |

What is left is no longer dominated by the runtime. The five largest entries are now
OpenCV (98 MB), SciPy (70 MB), onnxruntime (70 MB), NumPy (46 MB) and scikit-image (25 MB) —
the decoder's own dependencies, which neither ADR touches. Dropping Torch also took SymPy
(29 MB), networkx and torchgen with it, none of which the runtime ever called.

The Linux figures ADR 0004 recorded — 969 MB with the CPU pin, 4801 MB without it — have no
successor row, because there is no pin to get wrong: `onnxruntime` resolves the same CPU wheel
on every target platform. `tests/inference/integration/test_published_package.py` resolves the
built wheel's own metadata on all five targets with **no flag** and fails if any `nvidia-*`,
`triton`, or `torch` appears.

## Costs accepted

**Latency.** ONNX Runtime is slower on the forward pass — 40% on segment, 20% per transcribed
line, per ADR 0004's measurements. End to end that is 12–14% of the median, because the decoder
dominates and the decoder is the same NumPy in both. Accepted: a fifth of a second on a page is
not worth 475 MB and a 7 GB memory ceiling.

**A conversion step exists again.** Adding a model is now export-then-publish rather than
publish, and there is a class of defect — the graph drifting from what was trained — that
cannot occur when one framework does both. This is the real cost, it is exactly what ADR 0004
was right to dislike, and the accumulator bug above is a worked example of it. It is mitigated
by keeping the oracle in the repository: `tests/export/` runs both the graph and the artifact on
real weights and compares them, so drift is a test failure rather than a quiet accuracy loss.

**Torch is still in the repository.** It has to be, to trace the graph. So "the published
closure has no Torch" is a property of the import graph, not of the filesystem, and one
convenience import would undo it silently on a machine where Torch is installed.
`test_no_torch_remains_in_the_inference_import_graph` imports the package in a fresh interpreter
and asserts nothing Torch-shaped appears in `sys.modules`.

**`src/` is excluded from ruff**, so moving the export tree there means it is no longer linted.
That exclusion is a deliberate suppression of the research tree recorded elsewhere; the export
tree inherits it as a side effect rather than by intent, and `ruff check src/model/inference_export`
still reports on it.

## Consequences

- `uv tool install nomicous-inference` is the whole install command. There is no
  `--torch-backend=cpu`, no uv version floor for it to be honoured, and no 4.8 GB failure mode
  behind plain `pip`. `onnxruntime` publishes one CPU wheel per platform and there is no CUDA
  variant to resolve by accident.
- Intel macOS is supportable again.
- Peak RSS on a page returns to ~3.0 GB from ~7.0 GB.
- The **Hub artifact** is `.onnx` for both architectures. `find_hub_artifact` names only that
  suffix: `snapshot_download` fetches the whole revision, so every cache directory also holds
  the native checkpoint, and a preflight that accepted either would let directory contents pick
  the runtime.
- Checkpoint loading stops being a code-execution surface. ADR 0004 mitigated `torch.load` with
  `weights_only=True` and digest-before-load ordering; there is no unpickling left to reach,
  because the suffix check refuses a `.pt` outright. The digest check stays — it is integrity,
  not sandboxing.
- ADR 0004's amendment to ADR 0002 is itself reversed: the Torch modules leave the published
  package, as ADR 0002 originally required.
