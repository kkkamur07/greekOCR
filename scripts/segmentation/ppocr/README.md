# PP-OCR segmentation tooling

One-off operator tooling for the PP-OCRv6 detection parity check of
2026-09-19. These scripts run on the inference box, not in this repo's
environments, and they add no dependencies to any `pyproject.toml`.

## Scripts

| Script | Purpose |
|--------|---------|
| `export_onnx.py` | PaddleX medium-det directory → `pp-ocrv6-medium-det.onnx` with dynamic batch, height and width |
| `verify_parity.py` | Paddle static oracle vs exported ONNX over the same tensors, plus fixtures and timing |

`export_onnx.py` converts with paddle2onnx at a pinned opset and then asserts
the input axes are still symbolic: a frozen axis is refused rather than
written, because that exact failure has shipped before. It prints opset, input
and output names and shapes, file size and SHA-256, and saves the same facts
beside the artifact as `.meta.json`.

`verify_parity.py` is the measurement the export stands on. Per page at
max-side caps 1920 and 960 it builds the tensor with PaddleX 3.7.0's own
preprocessing classes, runs the same tensor through the Paddle static
predictor (mkldnn) and onnxruntime, compares the raw probability maps, runs
PaddleX's own DB postprocess on both maps, and compares the boxes. It also
runs the full `paddleocr.TextDetection` pipeline from the raw image (proving
the hand-built tensor path is faithful), checks five extra input shapes, saves
reference fixtures for the serving adapter, and times both runtimes. It exits
0 only on the full pass gate and non-zero otherwise.

## Runbook (on the box)

The reference environment is the benchmark tree; the parity venv lives beside
it. `system-libs` supplies the libgomp the Paddle wheels need:

```bash
cd /root/ppocrv6-onnx-parity-20260919
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
LD_LIBRARY_PATH=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu \
./venv/bin/python export_onnx.py \
  --model-dir /root/ppocrv6-bench-20260919/models/medium \
  --output /root/ppocrv6-onnx-parity-20260919/pp-ocrv6-medium-det.onnx
BASE="./venv/bin/python verify_parity.py --model-dir /root/ppocrv6-bench-20260919/models/medium \
  --onnx /root/ppocrv6-onnx-parity-20260919/pp-ocrv6-medium-det.onnx \
  --images /root/orli-bench/images \
  --grec-images /root/ppocrv6-bench-20260919/images \
  --workdir /root/ppocrv6-onnx-parity-20260919/run"
RUN="OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
LD_LIBRARY_PATH=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu"
$RUN $BASE maps > run/maps.log 2>&1
$RUN $BASE shapes > run/shapes.log 2>&1
$RUN $BASE pipeline > run/pipeline.log 2>&1
$RUN $BASE timing > run/timing.log 2>&1
$RUN $BASE report --report-json /root/ppocrv6-onnx-parity-20260919/parity-report.json > run/report.log 2>&1
```

Run the stages in order and check each log before moving on. `maps`
exits non-zero when its part of the pass gate fails; in that case stop and do
not run the later stages.

The ONNX itself is never committed to git; it stays in the artifact
directories. Full results, including the per-page parity table, the
shape-coverage table and timing, are in
`docs/inference/ppocrv6-onnx-parity-2026-09-19.md`.
