# Retired: the ONNX Runtime inference path

**Retired 2026-08-04** by [ADR 0004](../../docs/adr/0004-pytorch-is-the-inference-runtime.md),
implemented in issue 049. Nothing here is imported by the running system. It is
kept because the conversion work was non-trivial and rediscovering it from git
history is far harder than reading it here.

## Why it existed, and why it stopped

Local inference used to ship as a frozen native application: a four-platform
PyInstaller build, Developer ID notarization, Authenticode signing, and no
update channel. Two properties followed from that, and both were load-bearing:
every megabyte cost money and build time, and whatever dependency tree we froze
was the one researchers kept running. A small runtime was worth real complexity,
so we converted the trained PyTorch graphs to ONNX, published a second set of
artifacts, kept Torch out of the bundle by denylist, and maintained a parity
harness to prove the converted graph still matched the trained one.

ADR 0002 replaced the frozen bundle with `uv tool install` from PyPI. That
deleted both properties — there is no per-megabyte signing cost and there *is*
an update channel — and with them the reason to run two runtimes for one set of
models. Training was already PyTorch; making inference PyTorch too removes the
conversion step and the entire class of defect where what runs has drifted from
what was trained.

## What is in here

```
inference/architectures/calamari/onnx.py   ONNX Runtime transcribe adapter
inference/architectures/blla/onnx.py       ONNX Runtime segment adapter
inference/architectures/blla/numpy_support.py
                                           the Torch-free preprocessing and
                                           decoding helpers the BLLA ONNX
                                           adapter needed
export/calamari/export.py                  Calamari graph -> best.onnx
export/blla/export.py                      BLLA graph -> blla.onnx
export/blla/__init__.py
scripts/convert_calamari.py                CLI wrapper for the Calamari export
scripts/export_blla_onnx.py                CLI wrapper for the BLLA export
tests/test_calamari_onnx.py                export + runtime parity, Calamari
tests/test_blla_onnx.py                    export + runtime parity, BLLA
tests/test_blla_parity.py                  ONNX and native vs the Kraken oracle
```

### The conversion pipeline

Both architectures followed the same shape: load a native checkpoint into the
inference-owned PyTorch graph, trace it with `torch.onnx.export`, then reopen
the resulting file with the `onnx` package and stamp `metadata_props` with
everything the runtime would need but the graph could not carry.

**Calamari** (`export/calamari/export.py`). The graph is a CNN-BiLSTM with
`LazyBiLSTM` and `nn.LazyLinear` layers, so the exporter ran one dummy forward
pass to materialize them before loading the state dict. The traced artifact
carried a dynamic time axis and a static batch of 1 — the runtime submits one
line crop at a time, and a variable-batch LSTM state is not supported by the
tracer. Embedded metadata: `format=calamari-onnx-v1`, `classes`, `line_height`,
`blank_index`, `temperature`, and the JSON `charset`, which is how the ONNX
runtime got a codec without reading the `.pt`. The exporter *baked* any positive
temperature into the graph (`CalamariTorchModel.forward` divides the logits
before tracing), so the runtime validated the metadata value but deliberately
did not re-apply it.

**BLLA** (`export/blla/export.py`). This one carries the hardest-won piece of
knowledge in the archive, in `_ExportGroupNorm`. `nn.GroupNorm` lowers to
`Reshape([0, 32, -1]) -> InstanceNormalization`, which flattens a group into a
single axis of `C/G * H * W` elements; onnxruntime's CPU kernel then accumulates
that group's mean and variance in one serial float32 accumulator. On a
1800x2471 manuscript page that is 2,224,800 values through one accumulator, and
on real spatially-correlated post-ReLU activations the rounding bias
accumulates instead of cancelling — the recovered per-group sigma drifted ~1.2e-3
relative, logits moved by up to **0.89**, and the handful of pixels that crossed
the 0.5 sigmoid boundary restructured short line polygons entirely (IoU 0.50
against the oracle). The fix was a trace-only module that reduces over the width
axis first and then over the remaining group axis: identical arithmetic (every
block has exactly `width` elements), but no accumulator sees more than a few
thousand terms. It was applied *only* while tracing, because the staged
reduction perturbs the native float32 logits by up to 1.8e-3 — harmless
numerically, but enough to break the native decoder's bit-exact agreement with
the Kraken oracle.

That is the flavour of defect a conversion step introduces, and the reason the
parity harness existed.

### What the parity harness checked

`tests/test_blla_parity.py` ran three comparisons against Kraken's bundled
`blla.mlmodel` as the oracle, over every manuscript page available locally:

1. **Preprocessing** — our `preprocess_blla_image` tensor against Kraken's
   `ImageInputTransforms` output, `atol=1e-6`.
2. **Weights** — our `BLLATorchModel.state_dict()` keys and tensors against
   `oracle.nn.state_dict()`, exact equality.
3. **Logits and geometry** — native Torch vs oracle at `mean < 1e-5`,
   `max < 1e-4`; ONNX vs oracle at `mean < 2e-3`, `p99 < 2e-2`, `max < 0.2`;
   then decoded lines compared by count, reading order, baseline Hausdorff
   distance (<= 32 px) and boundary IoU (min >= 0.90, mean >= 0.95).

The two runtimes needed *different* tolerances against the same oracle. That
asymmetry is the cost of conversion, stated numerically.

It also pinned `test_torch_and_numpy_decoders_match_on_identical_real_logits`,
which fed one set of real logits through both decoder implementations to prove
that any ONNX drift came from the graph and not from the Torch-free decoder.

`tests/test_calamari_onnx.py` compared the exported session against the
reference Torch forward at `rtol=1e-4, atol=2e-4` across odd and even time
widths — odd widths matter because `same` padding is asymmetric and a tracer
that constant-folded the shape would pass at the trace width and silently fail
elsewhere.

## What was deleted rather than archived

These existed only to police Torch as a forbidden dependency, and Torch is now
the runtime:

- `packaging/helper/excludes.txt` — the PyInstaller denylist (`torch`,
  `torchgen`, `torchvision`, `safetensors`, `kraken`, plus the native
  architecture modules)
- `packaging/helper/scripts/verify-bundle.py` — the release-time verifier that
  walked the frozen bundle and both PyInstaller TOC manifests for those names
- the `parity` dependency group and its `kraken>=7.0.2` pin — with one runtime
  there is nothing to be at parity *with*
- `inference/jobs/runner.py`'s `onnx_only` flag and its native-artifact
  rejection, plus `tests/inference/unit/test_runner_onnx_only.py`
- the published `.onnx` artifacts in `src/hf/staging/` (dual-format publishing)

## What moved instead of being archived

The Torch graph the exporters imported was never ONNX-specific — it was the
trained model. Under ADR 0004 it *is* the runtime, so it moved into the
inference package rather than into this directory:

| was | is now |
|---|---|
| `src/model/inference_export/calamari/model.py` | `inference/architectures/calamari/model.py` |
| `src/model/inference_export/calamari/layers.py` | `inference/architectures/calamari/layers.py` |
| `src/model/inference_export/calamari/config.py` | `inference/architectures/calamari/config.py` |
| `load_calamari_checkpoint` (in `export.py`) | `inference/architectures/calamari/checkpoint.py` |

One export-only detail was dropped in that move: `layers.pad_same` carried a
branch for the legacy ONNX tracer, because `x.shape` becomes a Python constant
during tracing and temporal padding had to stay dynamic. It read:

```python
if torch.onnx.is_in_onnx_export():
    shape = torch._shape_as_tensor(x)
    time_size: int | Tensor = shape[-2]
    height_size: int | Tensor = shape[-1]
else:
    time_size = x.shape[-2]
    height_size = x.shape[-1]
```

with a matching `Tensor` branch in `_same_padding_amount`. Restore both if you
ever revive the exporter.

## Reviving this

You would need, roughly:

1. `onnxruntime` and `onnx` back in the dependency closure (they were
   `onnxruntime>=1.23.2,<1.24` in `[project].dependencies` and `onnx>=1.17.0`
   in an `export` dependency group).
2. `kraken>=7.0.2` back, if you want the parity harness — it is the oracle.
3. The graph modules above are imported from their new locations by the
   archived exporters; `numpy_support.py` here holds the pieces that were
   removed from `inference/architectures/blla/`.
4. Re-export the artifacts and re-pin `artifact_sha256` in
   `inference/registry.yaml`, which now points at the native `.pt` and
   `.safetensors` files at the same **Hub revision**.

Before you do: read ADR 0004's "Alternatives considered". Shipping both runtimes
and selecting at load was considered and rejected — that is the parity burden
made permanent and given a configuration surface.

## Measured at retirement

Single-page CPU latency and output comparability were recorded during issue 049,
on a real 3400 px manuscript page and the published weights. The summary is in
ADR 0004 under "Costs accepted". Headline: decoded output was identical, and
PyTorch was *faster* than ONNX Runtime on both architectures — but BLLA's peak
RSS more than doubled.
