# PP-OCRv6 medium detection ONNX parity, 2026-09-19

## What was exported

* Source model: `PaddlePaddle/PP-OCRv6_medium_det` (PaddleX model directory on the
  box at `/root/ppocrv6-bench-20260919/models/medium`), pinned at Hub revision
  `8e0f56fb2ef86b461d99cfc7ac5c137738985f61`
  (files `inference.json`, `inference.pdiparams`, `inference.yml`).
* Artifact: `/root/ppocrv6-onnx-parity-20260919/pp-ocrv6-medium-det.onnx`
  (local copy at `_ppocr-parity/pp-ocrv6-medium-det.onnx`).
* ONNX opset: 21 (`ai.onnx`). The file loads and runs in onnxruntime 1.30.0,
  the version the production worker runs. No trial history is recorded here,
  only the final opset and the fact that it loads.
* SHA-256: `09e4c827c5bb20a0344374bbf8b88d41b7c8bf2be3a0db82ffed3bf090eacfe3`
  (identical on the box and in the local artifact directory).
* Size: 62064678 bytes.
* Input: `x`, shape `[N, 3, H, W]`, all of batch, height and width dynamic.
* Output: `fetch_name_0`, shape `[N, 1, H, W]`, all dynamic.
* `onnx.checker.check_model` passes at export time (see `export_onnx.py`).
* Export environment (frozen in `_ppocr-parity/freeze.txt`):
  `paddlepaddle==3.2.2`, `paddle2onnx==2.1.0`, `onnx==1.17.0`,
  `onnxruntime==1.30.0`, `paddleocr==3.7.0`, `paddlex==3.7.0`, `numpy==2.3.5`.

## Exact preprocessing (hand-built tensor path)

The verification builds the input tensor with PaddleX 3.7.0's own
preprocessing classes, driven by the model's `inference.yml`:

1. `DecodeImage`: BGR, `channel_first` false.
2. `DetResizeForTest` with `limit_side_len` equal to the cap (1920 or 960),
   `limit_type="max"`, `input_shape=None`. The resized height and width are
   rounded to multiples of 32.
3. `NormalizeImage`: scale `1./255.`, mean `[0.485, 0.456, 0.406]`,
   std `[0.229, 0.224, 0.225]`, order `hwc`.
4. `ToCHWImage`, then `ToBatch`, cast to float32.

The DB postprocess is PaddleX's own `DBPostProcess` Python code, unchanged:
`thresh=0.2`, `box_thresh=0.45`, `max_candidates=3000` (from `inference.yml`),
`unclip_ratio=1.4`, `use_dilation=False`, `score_mode="fast"`, `box_type="quad"`.
The same postprocess runs on both the Paddle and the ONNX probability maps,
so any box difference is caused by the runtimes, not by the postprocess.

## Per-page parity (same tensor, Paddle static vs ONNX)

Reference: Paddle static predictor (mkldnn, 4 CPU threads) from the benchmark
venv recipe. Challenger: onnxruntime 1.30.0 `CPUExecutionProvider`
(4 intra-op threads). One line per page per cap; boxes are counted after the
shared postprocess, corner distance is in original-image pixels.

Cap 1920:

| page | paddle boxes | onnx boxes | delta | max corner px | max abs prob diff | mean abs prob diff | binarisation diff px |
|------|--------------|------------|-------|---------------|-------------------|--------------------|----------------------|
| vat-1r | 78 | 78 | 0 | 0.000 | 4.61e-04 | 1.75e-06 | 3 |
| vat-1v | 71 | 71 | 0 | 0.000 | 3.07e-04 | 7.25e-07 | 0 |
| vat-2r | 69 | 69 | 0 | 0.000 | 5.23e-04 | 1.41e-06 | 0 |
| vat-2v | 64 | 64 | 0 | 0.000 | 2.35e-04 | 6.67e-07 | 1 |
| vat-3r | 69 | 69 | 0 | 0.000 | 7.69e-04 | 1.91e-06 | 4 |
| vat-7v | 67 | 67 | 0 | 0.000 | 1.08e-03 | 4.12e-06 | 4 |
| c10 | 65 | 65 | 0 | 0.000 | 5.84e-04 | 1.70e-06 | 1 |
| c11 | 131 | 131 | 0 | 0.000 | 2.85e-04 | 9.89e-07 | 2 |
| c12 | 130 | 130 | 0 | 0.000 | 1.83e-04 | 7.25e-07 | 1 |
| c13 | 135 | 135 | 0 | 0.000 | 3.36e-04 | 1.21e-06 | 1 |
| c14 | 138 | 138 | 0 | 0.000 | 1.91e-04 | 1.24e-06 | 3 |
| c21 | 130 | 130 | 0 | 0.000 | 2.67e-04 | 2.41e-06 | 6 |
| grec-p1 | 29 | 29 | 0 | 0.000 | 2.58e-04 | 6.39e-07 | 0 |
| grec-p4 | 57 | 57 | 0 | 0.000 | 6.70e-04 | 1.75e-06 | 3 |

Cap 960:

| page | paddle boxes | onnx boxes | delta | max corner px | max abs prob diff | mean abs prob diff | binarisation diff px |
|------|--------------|------------|-------|---------------|-------------------|--------------------|----------------------|
| vat-1r | 72 | 72 | 0 | 0.000 | 2.74e-04 | 1.29e-06 | 5 |
| vat-1v | 71 | 71 | 0 | 0.000 | 2.79e-04 | 1.42e-06 | 1 |
| vat-2r | 66 | 66 | 0 | 0.000 | 2.53e-04 | 1.05e-06 | 0 |
| vat-2v | 62 | 62 | 0 | 0.000 | 2.28e-04 | 1.20e-06 | 0 |
| vat-3r | 66 | 66 | 0 | 0.000 | 1.99e-04 | 9.12e-07 | 1 |
| vat-7v | 70 | 70 | 0 | 0.000 | 2.83e-04 | 1.95e-06 | 0 |
| c10 | 65 | 65 | 0 | 0.000 | 5.84e-04 | 1.70e-06 | 1 |
| c11 | 134 | 134 | 0 | 0.000 | 3.64e-04 | 3.00e-06 | 2 |
| c12 | 132 | 132 | 0 | 0.000 | 1.95e-04 | 1.26e-06 | 1 |
| c13 | 132 | 132 | 0 | 0.000 | 1.06e-04 | 7.37e-07 | 0 |
| c14 | 136 | 136 | 0 | 0.000 | 1.88e-04 | 9.81e-07 | 0 |
| c21 | 128 | 128 | 0 | 0.000 | 1.49e-04 | 1.28e-06 | 1 |
| grec-p1 | 27 | 27 | 0 | 0.000 | 2.64e-04 | 1.08e-06 | 1 |
| grec-p4 | 55 | 55 | 0 | 0.000 | 1.35e-04 | 7.27e-07 | 0 |

Worst page per cap: cap 1920 is vat-7v (max abs prob diff 1.08e-03, still
below the 1e-2 finding threshold); cap 960 is c10 (5.84e-04). Every case has
box delta 0 and corner distance 0.000 px, so the pass gate holds on all
28 cases. The raw stage output is `_ppocr-parity/maps.json` and
`_ppocr-parity/maps.log`.

The raw Paddle and ONNX probability maps for `c13` and `grec-p1` at cap 1920
are saved as `.npy` pairs (`c13.paddle.npy`, `c13.onnx.npy`,
`grec-p1.paddle.npy`, `grec-p1.onnx.npy`) in `_ppocr-parity/`.

## Pipeline check (hand-built tensor path vs full TextDetection pipeline)

For every page at both caps, the hand-built tensor path gives exactly the same
boxes as the full `paddleocr.TextDetection` pipeline
(`engine="paddle_static"`, mkldnn, 4 threads): `hand_delta=0` and
`hand_corner=0.000px` in all 28 cases (`_ppocr-parity/pipeline.json`). This
proves the preprocessing copy in `verify_parity.py` is faithful.

Reference fixtures for the serving adapter (from the full Paddle pipeline at
cap 1920, with image name, image sha256, original size, resized tensor shape,
boxes with scores) are saved per page in `_ppocr-parity/fixtures/` (14 files).

## Shape coverage

Five extra input sizes on synthetic tensors, Paddle static vs ONNX:

| label | input (NCHW) | max abs prob diff | onnx output shape ok |
|-------|--------------|-------------------|----------------------|
| 320x320 | 1x3x320x320 | 1.26e-07 | True |
| wide-1920x320 | 1x3x320x1920 | 1.48e-07 | True |
| tall-640x1920 | 1x3x1920x640 | 2.24e-07 | True |
| odd-480x736 | 1x3x736x480 | 1.55e-07 | True |
| odd-1056x864 | 1x3x864x1056 | 1.60e-07 | True |

All five return the full input resolution with negligible numeric difference,
so no axis is frozen at the trace shape. Raw output: `_ppocr-parity/shapes.json`.

## Timing (detector plus postprocess per page at cap 1920)

Median of 3 timed runs after 1 warmup, per runtime, on 3 pages:

| page | 8 threads paddle (s) | 8 threads onnx (s) | 2 threads paddle (s) | 2 threads onnx (s) |
|------|----------------------|--------------------|----------------------|--------------------|
| c13 | 1.634 | 6.899 | 3.797 | 20.260 |
| vat-1r | 1.966 | 7.883 | 4.982 | 23.159 |
| grec-p1 | 1.934 | 7.786 | 4.724 | 22.757 |

The ONNX on CPU is about 4x slower than Paddle mkldnn at 8 threads and about
5x slower at 2 threads. Raw output: `_ppocr-parity/timing.json`.

## How to reproduce (on the box)

```bash
cd /root/ppocrv6-onnx-parity-20260919
LIBS=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LIBS
BASE="./venv/bin/python verify_parity.py \
  --model-dir /root/ppocrv6-bench-20260919/models/medium \
  --onnx /root/ppocrv6-onnx-parity-20260919/pp-ocrv6-medium-det.onnx \
  --images /root/orli-bench/images \
  --grec-images /root/ppocrv6-bench-20260919/images \
  --workdir /root/ppocrv6-onnx-parity-20260919/run"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE maps > run/maps.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE shapes > run/shapes.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE pipeline > run/pipeline.log 2>&1
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $BASE timing > run/timing.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $BASE report \
  --report-json /root/ppocrv6-onnx-parity-20260919/parity-report.json > run/report.log 2>&1
```

The `report` stage merges the stage JSON files, applies the pass gate
(zero box delta, max corner distance at most 1.0 px, max abs prob diff at
most 1e-2 on every page at both caps, plus the shape and pipeline checks),
and exits non-zero on failure. It exited 0 with `failures=[]`.

## Limitations

* The ONNX is the medium detection model only. Recognition and the rest of
  the OCR pipeline are not covered.
* Numerics are measured on the 14 benchmark pages (plus 5 synthetic shapes),
  not on arbitrary inputs.
* Timing covers detector plus DB postprocess on CPU only; no GPU was involved.
* The ONNX is markedly slower than Paddle mkldnn on this CPU box (see Timing),
  so serving cost needs its own decision.
* No opset trial history is recorded; only the final opset 21 and the fact
  that it loads in onnxruntime 1.30.0 are verified facts.
* The ONNX itself is not committed to git; it lives in the box work directory
  and the local artifact directory `_ppocr-parity/`.
