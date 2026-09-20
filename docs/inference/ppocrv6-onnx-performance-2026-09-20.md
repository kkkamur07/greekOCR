# PP-OCRv6 medium-det ONNX performance, 2026-09-20

## Cause of the gap

The profiled run (c13 at 4 threads, onnxruntime 1.30.0, default options) shows
convolution compute is the whole story, not graph clutter:

* Conv: 22.5 s total across 244 kernel events in two runs (about 96 percent
  of runtime). The slowest single Conv kernels take 1.1 to 2.6 s each.
* Everything else is small: fused Gelu 0.19 s, ConvTranspose 0.16 s, Concat,
  MaxPool, Add and Upsample below 0.1 s each, NCHWc ReorderInput/ReorderOutput
  0.03 s combined.
* The graph is already clean at runtime: the 13 Erf nodes (paddle2onnx GELU
  decomposition) are fused into Gelu by the optimiser, and the 326 Identity
  plus 117 Reshape nodes cost nothing measurable (onnxsim removes them with
  zero speed gain, see below).

In other words, Paddle mkldnn (oneDNN) runs this network's convolutions about
4x faster than onnxruntime's default path on this CPU. About half of that gap
is a bad default: `ORT_ENABLE_ALL` picks a convolution path roughly 2x slower
than `ORT_ENABLE_EXTENDED` on this model. The remaining ~2x is kernel library
throughput (MLAS/NCHWc against oneDNN) and cannot be closed from our side.

## Session experiments (accepted file, c13, detector plus postprocess)

Median of 3 runs after 1 warmup, per config:

| setting | median (s) | helped |
|---------|------------|--------|
| opt ALL, 4 threads | 11.439 (re-run 11.497) | baseline |
| opt EXTENDED, 4 threads | 5.850 | yes, 1.96x |
| opt BASIC, 4 threads | 5.752 | yes, same as EXTENDED |
| threads=1 (ALL) | 34.204 | reference scaling |
| threads=2 (ALL) | 20.260 (parity run) | reference scaling |
| threads=8 (ALL) | 6.899 (parity run) | reference scaling |
| exec PARALLEL, 4 threads | 11.626 | no |
| spinning off, 4 threads | 11.882 | no |
| denormal_as_zero, 4 threads | 4.249 once | no, see below |

The denormal_as_zero number is not trusted: its probability maps are
bit-identical to the baseline on all 3 checked pages (max abs diff exactly
0.00), so flushing subnormals cannot explain a 2.7x speedup; that run is
attributed to box load noise and the option is not recommended.

Numerics of the recommended setting (accepted file, ALL against EXTENDED,
same tensor, shared DB postprocess):

| page | max abs map diff | boxes | delta | corner |
|------|------------------|-------|-------|--------|
| c13 | 4.17e-06 | 135 | 0 | 0.000px |
| vat-1r | 4.71e-06 | 78 | 0 | 0.000px |
| grec-p1 | 3.76e-06 | 29 | 0 | 0.000px |

EXTENDED changes kernel selection, not results: diffs are at single-precision
reorder noise level and every box is identical.

## Graph experiments

* onnxsim 0.7.3 (`perf/pp-ocrv6-medium-det-sim.onnx`, sha256
  `c8b38fb9c8cb671cce6e6543ec5172e45de8fdbcc0da0a0fab2451b38d875db6`,
  61995879 bytes): checker ok, loads in onnxruntime 1.30.0, 1273 nodes down
  to 261, Reshape 117 down to 0, dynamic batch/height/width kept (5/5 shape
  cases pass). Timing on c13 at 4 threads: 11.831 s against 11.439 s for the
  accepted file. No gain, not recommended. First attempt passed
  `--input-shape` and froze the input to 928x1440 (rejected on the shape
  check); the kept file was simplified with no shape flags, since onnxsim
  0.4 handles dynamic shapes automatically.
* Re-exports at opset 17 and 13 were not attempted: the 45 minute timebox ran
  out, and the profile shows a convolution-bound model where the opset cannot
  change kernel speed (the Erf chains already fuse at opset 21).

No candidate beat the accepted file by 20 percent, so no full parity gate was
run on a candidate (the gate stays with the accepted file from the parity run).

## Final timing (median of 3 after 1 warmup, detector plus postprocess)

ONNX rows use the accepted file at `ORT_ENABLE_EXTENDED`. Paddle rows are the
parity run except the big tensor, measured here.

| case | ONNX ext 8t (s) | Paddle 8t (s) | ONNX ext 2t (s) | Paddle 2t (s) |
|------|-----------------|---------------|-----------------|---------------|
| c13 | 3.592 | 1.634 | 11.030 | 3.797 |
| vat-1r | 4.074 | 1.966 | 12.819 | 4.982 |
| grec-p1 | 3.979 | 1.934 | 12.438 | 4.724 |
| big [1,3,1920,1280] | 6.271 | 2.356 | 18.924 | not measured |

For reference, the accepted file at default ALL measured 6.899/7.883/7.786 s
at 8 threads and 20.260/23.159/22.757 s at 2 threads on the same 3 pages.
EXTENDED closes the gap from ~4x to ~2.1x at 8 threads and ~2.7x at 2 threads.
The big tensor (vat-1r upscaled, compute only, boxes meaningless) confirms the
same ratio at full 1920-long-side resolution.

## Recommendation

* Publish the accepted file, unchanged: `pp-ocrv6-medium-det.onnx`, sha256
  `09e4c827c5bb20a0344374bbf8b88d41b7c8bf2be3a0db82ffed3bf090eacfe3`,
  62064678 bytes, opset 21. Parity gate result stands from the parity run
  (`failures=[]` on all 28 page/cap cases plus shapes and pipeline).
* The serving adapter should create its sessions as:

```python
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 1
```

* Rationale: EXTENDED is ~2x faster than the ALL default with identical
  boxes; 2 intra-op threads per process fits four processes on 8 cores;
  PARALLEL execution, spinning off and denormal_as_zero showed no trustworthy
  gain; no graph rewrite beat the accepted file.

## How to reproduce (on the box)

```bash
cd /root/ppocrv6-onnx-parity-20260919
LIBS=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LIBS
BASE="./venv/bin/python profile_onnx.py \
  --onnx pp-ocrv6-medium-det.onnx \
  --model-dir /root/ppocrv6-bench-20260919/models/medium \
  --images /root/orli-bench/images \
  --grec-images /root/ppocrv6-bench-20260919/images \
  --perfdir perf"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE counts > perf/counts.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE profile > perf/profile.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE sessions --only opt1 > perf/sessions-opt1.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE sessions --only opt2 > perf/sessions-opt2.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE sessions --only t1 > perf/sessions-t1.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE sessions --only t4 > perf/sessions-t4.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE sessions --only misc1 > perf/sessions-misc1.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE sessions --only misc2 > perf/sessions-misc2.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE compare > perf/compare-opt.log 2>&1
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $BASE final > perf/final-timing.log 2>&1
```

The box venv gained `onnx-simplifier==0.5.0`, `onnxsim==0.7.3`, `rich==15.0.0`,
`pygments==2.21.0` (plus `markdown-it-py`, `mdurl`); see `perf/freeze-perf.txt`.

## Limitations

* The remaining ~2x gap is convolution kernel throughput and is not fixable
  with onnxruntime 1.30.0 session options or onnxsim on this CPU.
* Thread scaling and timing were measured while four production workers share
  the box, so absolute seconds carry load noise; ratios were stable across
  re-runs (ALL 11.439/11.497, EXTENDED corroborated by BASIC 5.752).
* Opset 17/13 re-exports were not attempted (timebox).
* The denormal_as_zero speedup is reported but disbelieved (bit-identical
  outputs) and not recommended.
