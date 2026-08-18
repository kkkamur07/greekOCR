# 0004. PyTorch is the inference runtime; ONNX is archived

- Status: Superseded by [0006](./0006-onnx-runtime-is-the-inference-runtime.md) (2026-08-05)
- Date: 2026-08-04
- Builds on: [0002](./0002-inference-cli-replaces-loopback-helper.md), which moved distribution
  from frozen native installers to `uv tool install` from PyPI.

## Context

ONNX Runtime was chosen when local inference shipped as a frozen native application. In that
world two properties were load-bearing:

- Every megabyte shipped through a four-platform build, Developer ID notarization, and
  Authenticode signing.
- There was no update channel, so the vendored dependency tree we distributed was the one
  researchers kept running.

Both properties made a small, dependency-light runtime worth real complexity: an export
pipeline, a second set of published artifacts, a `Torch`-exclusion denylist (`excludes.txt`), a
bundle verifier, and a parity apparatus to prove the converted graph still matched the trained
one.

ADR 0002 deleted the frozen bundle. Its own packaging section, however, carried the old
constraint forward unexamined. It required Torch to be moved *out* of the published package and
treated the denylist's removal as the win. That was the bundle-era premise surviving its own
justification.

## Decision

PyTorch is the inference runtime. ONNX Runtime, its conversion scripts, and the ONNX↔Torch
parity tooling are archived under `archive/onnx-runtime/`.

Archived rather than deleted, because git history is a poor place to find a retired subsystem,
and the archive carries a README naming what it was and why it was retired.

## Rationale

The constraint that paid for ONNX is gone. `uv tool install` has neither property that made a
lean runtime worth an export pipeline: there is no signing cost per megabyte, and there *is* an
update channel, enforced by the version floor in ADR 0002. The reasoning that justified ONNX
did not survive the change in distribution, and keeping the conclusion after deleting its
premise is how architectures accumulate cost nobody can account for.

Training is already PyTorch. One framework end to end removes the conversion step, and with it
the entire class of defect where what runs has drifted from what was trained. That is what the
parity apparatus existed to detect, and with one runtime there is nothing to be at parity
*with*.

The Torch graphs already exist and already run, so this is a deletion rather than a port:

|                                                                               | lines |
| ----------------------------------------------------------------------------- | ----- |
| `src/model/inference_export/calamari/{model,layers,config}.py`                | 279   |
| `inference/architectures/blla/blla_model.py`                                  | 142   |
| `inference/architectures/calamari/{model,layers,config}.py` (re-export shims) | 41    |

(Implemented in 049: the Calamari graph moved into `inference/architectures/calamari/`,
replacing the shims, and its checkpoint loader became
`inference/architectures/calamari/checkpoint.py`.)

`blla.py` already loads a state dict and calls `.eval()`. The work is removing the ONNX loaders
and pointing the runtime at graphs that are present today.

Native checkpoints are already published. `best.pt` and `blla.safetensors` sit beside the
`.onnx` artifacts in the staging tree, and the offline bundled weights are `.pt` only. There is
no re-publishing and no new artifact SHA-256 story, because the digests move to the artifacts
already there.

Several things also stop having a reason to exist: `excludes.txt`, the bundle verifier, the
`kraken` parity dependency, and dual-format artifact publishing. ADR 0002 wanted the first two
dead via a package boundary, and this kills them by construction instead, because there is no
forbidden dependency left to police.

## Costs accepted

Install size. Measured on the development machine, `onnxruntime` is 63 MB installed against
`torch` at 388 MB. Accepted as a one-time cost, since `uv` caches and a version bump of the
published package does not re-fetch Torch unless the Torch pin itself moves. The package must
pin the CPU-only build, because CUDA wheels dominate the download and are useless on a
researcher's laptop.

CPU latency, accepted without measurement and then measured. ONNX Runtime generally outperforms
PyTorch eager on the small convolutional and recurrent graphs these models use, and for local
inference that latency *is* the product experience. A benchmark was commissioned and stopped
before producing numbers, so this risk was accepted unmeasured rather than assessed, with the
implementing issue (049) required to record single-page CPU latency before and after.

It was measured during 049, and the expectation was wrong: PyTorch is faster on both
architectures. The mitigation held in reserve, `torch.compile` or a compiled runtime for
transcription only, is not needed.

See "Measured on the switch" below. What the measurement *did* surface is a memory cost that
was not anticipated at all, recorded there.

Checkpoint loading becomes a code-execution surface. Loading a pickled Torch checkpoint
executes code in a way loading an ONNX graph does not. This is mitigated rather than
eliminated, by preferring `safetensors`, keeping artifact SHA-256 verification ahead of the
load, and never unpickling an unverified file.

## Measured on the switch

Recorded during issue 049 on the development machine (Apple M-series, macOS 26, 10 cores), both
runtimes pinned to 4 intra-op and 1 inter-op threads, on a 3400 px manuscript page and the
published weights. Seven timed iterations after two warm-ups, with model load timed separately
on a cold cache. Transcribe latency is a whole page, meaning the 35 line crops BLLA found on
that page.

### Latency

|                              | ONNX Runtime 1.23.2        | PyTorch 2.10.0    | change     |
| ---------------------------- | -------------------------- | ----------------- | ---------- |
| Segment page, end to end     | 7.81 s median / 7.93 s p90 | 6.74 s / 10.10 s  | −14% median |
| Segment, model forward only  | 2.72 s / 2.76 s            | 1.62 s / 2.63 s   | −40% median |
| Segment, model load          | 0.016 s                    | 0.017 s           |            |
| Transcribe page (35 lines)   | 0.249 s / 0.273 s          | 0.219 s / 0.222 s | −12% median |
| Transcribe, per line         | 9.4 ms / 14.8 ms           | 7.5 ms / 8.5 ms   | −20% median |
| Transcribe, model load       | 0.038 s                    | 0.009 s           |            |

Segment latency is dominated by the NumPy and scikit-image decoder, at roughly 5.0 s on both
runtimes, which the runtime choice does not touch. The forward pass is the only part that
changed, and it is where PyTorch's 40% win lands. The one number that got worse is segment p90,
and it is tail noise on an unquiet laptop rather than a distribution shift, since min and
median moved down together.

### Memory, the unanticipated cost

| peak RSS   | ONNX Runtime | PyTorch |
| ---------- | ------------ | ------- |
| Segment    | 3.0 GB       | 7.0 GB  |
| Transcribe | 149 MB       | 328 MB  |

Segmenting one page costs 2.3 times the peak memory it used to. The page tensor is
`1×3×1800×2471` float32 and the graph is fully convolutional, so the intermediates are large in
either runtime, but PyTorch eager materializes them where ONNX Runtime reuses a planned arena.
7 GB is a real constraint on an 8 GB laptop and is the sharpest edge of this decision. It is
not a reason to reverse it. It is a reason to bound the input, and
`blla_preprocessing.MAX_WIDTH_TO_HEIGHT_RATIO` is the existing lever.

### Install size and cold-install time, measured on publication

"Install size" above was one dependency's installed size on a development machine. What a
researcher actually pays is the whole closure of the published package, and it was measured
when that package first existed (issue 050), from an empty `uv` cache, on a fast connection.
`nomikos-inference` and its twelve declared dependencies:

|                                        | installed | bytes fetched | cold install |
| -------------------------------------- | --------- | ------------- | ------------ |
| Linux aarch64, `--torch-backend=cpu`   | 969 MB    | 979 MB        | 7-8 s        |
| Linux aarch64, no flag                 | 4801 MB   | 4812 MB       | 71 s         |
| macOS arm64 (PyPI wheel is CPU-only)   | 811 MB    | 836 MB        | 7 s          |

Measured with `uv pip install` into an empty virtual environment and confirmed end to end with
`uv tool install --torch-backend=cpu`, which is the command a researcher runs.

Torch is 475-597 MB of that, and OpenCV, SciPy, NumPy, SymPy and scikit-image are most of the
rest. Seconds is a property of the connection rather than of the package. The number that
transfers is the roughly 980 MB, which is about 22 minutes on a 6 Mbit/s line. That is the
first-run experience this record said should be measured rather than assumed.

The CPU-only pin cannot be expressed in package metadata. No PEP 621 field names an index, and
`tool.uv.sources` is ignored when a package is consumed from a registry, so it pins this
repository's lock and nothing downstream. A plain resolve on either Linux architecture
therefore pulls sixteen `nvidia-*` and `triton` wheels behind `torch`, which is the 4801 MB
row. The pin is restated at install time instead, and the documented command is
`uv tool install nomikos-inference --torch-backend=cpu`.
`tests/inference/integration/test_published_package.py` resolves the wheel's own metadata on
all four target platforms and fails if any of them admits a CUDA wheel.

Two residual risks, neither solved. A researcher who installs with plain `pip` gets the 4.8 GB
tree and nothing stops them. And `uv tool install` only accepts `--torch-backend` from uv 0.10
onward, where uv 0.7.x reads `UV_TORCH_BACKEND=cpu` from the environment and silently installs
the CUDA tree anyway. The failure mode of a stale uv is therefore a quiet 4.8 GB download
rather than an error, which is the worst shape it could have taken.

Intel macOS drops off the supported set. PyTorch publishes no `x86_64-apple-darwin` wheel from
2.10 onward, so the package cannot be installed there at all. Under ONNX Runtime it could have
been.

### Output comparability

Decoded output is identical, not merely close. On the same page and the same weights, PyTorch
and ONNX Runtime produced the same 35 lines with byte-identical baselines, polygons, ceilings,
and external ids, and identical transcribed text for all 35 crops (627 characters). The ONNX
conversion was not lossy at the level that reaches a user.

Underneath, the two runtimes are not bit-identical, as expected. BLLA logits differ by 1.1e-05
mean and 9.2e-03 max, and no pixel crosses the 0.5 or 0.17 sigmoid thresholds the decoder uses.
Calamari character confidences differ by at most 2.6e-06. That agreement was bought rather than
free: see the `_ExportGroupNorm` note in `archive/onnx-runtime/README.md` for the accumulator
bug that had to be fixed in the exporter to reach it, which is itself an argument for having
one runtime.

## Consequences

- The published package carries a large dependency. Cold-install time on a poor connection
  becomes a real part of the first-run experience, measured on publication at roughly 980 MB
  fetched and 969 MB installed. See "Install size and cold-install time" above.
- The export tree stops being a build stage and becomes archived reference.
- ADR 0002's "One published package, not two" section is amended. The Torch modules stay inside
  the published package rather than moving out, and the denylist dies by construction rather
  than by boundary.
- Security patching keeps the shape ADR 0002 gave it, a dependency bump plus a version-floor
  bump, but the surface is now larger, since Torch and its transitive tree are in it.
- Peak memory for segmentation more than doubled. Machines that could segment a large page
  before may not be able to now, and the input bound in `blla_preprocessing` is the control
  surface if that turns into a real report.

## Alternatives considered

Keep ONNX Runtime. Rejected, because its justification was the frozen bundle, which no longer
exists. Retaining it means maintaining an export pipeline, a second artifact format, a
denylist, and a parity harness to serve a size constraint nothing now imposes.

Ship both runtimes and select at load. Rejected, because this is precisely the parity burden,
made permanent and given a configuration surface. Two runtimes for one set of models is the
cost being removed, not a compromise between the options.

Delete the ONNX code outright rather than archiving it. Rejected on the owner's instruction,
and reasonably so: the conversion work is non-trivial, and rediscovering it from history is far
harder than reading it from `archive/`.
