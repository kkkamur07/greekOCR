# PP-OCRv6 Syriac recognition ONNX parity, 2026-09-20

## What was exported

* Source model: `/Users/krishuagarwal/Desktop/Programming/python/greekOCR/ppocr-syriac.safetensors`
  (kraken safetensors, one model in file).
* Class: `kraken.lib.ppocr.PPOCRv6Model` (kraken 7.1.1), variant `medium`,
  input `(1, 3, 96, 0)`, 15,857,527 parameters, 1623 classes with class 0 the
  CTC blank, codec of 1622 entries, `seg_type` baselines. Network: PPLCNetV4
  backbone, LightSVTR neck, CTC head; `forward(x, seq_lens)` returns
  `(logits[N, 1623, 1, W/8], out_lens)` with no softmax inside.
* Artifact: `_ppocr-rec-env/artifacts/syriac-ppocr-v1/model.onnx`
  (outside the repo, never committed).
* SHA-256 source: `2545eca289525c534c55f96f9941f6eddccae059cfa6fd4bf0042a2a7063e17e`
* SHA-256 ONNX: `a839d951ea443f3fb04acbbde88ad263b984f39a4e50b49adb78e9ad742c8a22`
* ONNX file size: 63,524,245 bytes (single file, weights embedded).
* ONNX opset: 17. Loads and runs in onnxruntime 1.28.0, the version the
  inference server pins (`pyproject.toml`, `onnxruntime>=1.23.2`, locked at
  1.28.0).
* Export environment (`uv pip freeze`): `kraken==7.1.1`, `torch==2.14.0`,
  `onnx==1.23.0`, `onnxscript==0.7.2`, `onnxruntime==1.28.0`, `numpy==2.4.6`,
  `torchvision==0.29.0` (full freeze in the scout report).

## Contract (`ppocr-rec-onnx-v1`, architecture `ppocr_rec`)

* One input `image`, float32 NCHW `[1, 3, 96, W]`, already preprocessed by the
  kraken recipe (RGB, fixed-height resize to 96, horizontal white padding,
  scaled to [0, 1], inverted). Batch is static 1. Only axis 3 is dynamic,
  named `width`. There is no `seq_lens` input.
* One output `logits`, float32 `[1, T, 1623]`, time on axis 1, class 0 blank.
  Raw logits: no softmax, no temperature in the graph.
* Time rule: nominal subsampling is 8, but SAME-padded strided convolutions
  round some widths up, so `T = W // 8` does NOT hold everywhere (192 of the
  508 widths 5..512 give `W // 8 + 1`, starting at width 5). The exact rule,
  verified against Torch for every integer width 5..512 (5..7 spot-checked,
  8..512 exhaustive) plus the sweep widths
  below with zero mismatches, is `T = (((W + 1) // 2 + 1) // 2) // 2`, stamped
  as `time_formula` in `metadata_props` with `subsampling` kept as 8.
* `metadata_props` keys: `format`, `architecture`, `variant`, `input_layout`
  (NCHW), `input_name`, `output_names` (JSON), `input_channels` (3),
  `line_height` (96), `subsampling` (8), `time_formula`, `classes` (1623),
  `blank_index` (0), `charset` (JSON, see below), `pad` (16),
  `pad_fill` (255), `temperature` (1.0), `seg_type` (baselines),
  `preprocessing` (one sentence), `opset_version`, `source_format`
  (kraken-safetensors), `source_sha256`, `kraken_version`, `torch_version`.
  `pad` 16 and `temperature` 1.0 are kraken's inference defaults, read out of
  `RecognitionInferenceConfig` (`kraken/configs/base.py`, padding default at
  line 231, temperature default at line 228).

## Charset convention

Same indexing convention as the Calamari `charset` metadata: a JSON list of
length `classes` where index 0 is the CTC blank (empty string) and index `i`
is the grapheme with kraken label `i` (kraken labels are 1-indexed,
`kraken/lib/codec.py`, `PytorchCodec`). A grapheme may hold several Unicode
codepoints and still occupies one index; a codec entry spanning several labels
has no single index and the exporter rejects it rather than dropping it. The
Syriac checkpoint needs no rejection: all 1622 entries are single-label,
labels run contiguously 1..1622, every grapheme is one codepoint, and the
round trip (charset back to `c2l`) is exact.

## Exporter attempts

1. Legacy `torch.onnx.export(dynamo=False)`, opset 17 (house default): failed
   at first on `PPLCNetV4.forward` (`kraken/lib/ppocr/backbone.py`), which
   pools with `kernel_size=(h, 2)` where `h` is read off the feature map. A
   traced shape value cannot serve as a kernel size. Fixed by probing `h`
   once (6 at height 96) and freezing that integer on a deepcopy of the model
   (the `_with_export_group_norm` precedent in `export/blla/export.py`); the
   patch is bitwise-exact (0.0 max abs diff vs the original on noise input)
   because `h` depends only on the fixed input height, never on width. Export
   then succeeds. Chosen.
2. Legacy opset 18, with and without `do_constant_folding`: exports fine and
   agrees with the opset-17 artifact to 4 significant figures on every test
   input, so it buys nothing over the house default. Rejected (no benefit).
3. Dynamo exporter (`torch.onnx.export` dynamo path, opset 18): exports, but
   writes weights to a separate `.onnx.data` file (63 MB beside a 1 MB graph,
   against the single-file artifact convention), names the time axis with a
   computed expression instead of `time`, needs `torch.export` shape
   constraints the backbone guards do not satisfy out of the box, and agrees
   numerically no better than the legacy artifact. Rejected (worse artifact,
   same numerics).

## Exact commands

Export (from the `ppocr-rec` worktree):

```
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 $PY scripts/hf/export_ppocr_rec_onnx.py \
  --checkpoint $MAIN/ppocr-syriac.safetensors \
  --destination $ART/model.onnx \
  --report-json $ART/export-report.json
```

Parity gate (all 18 Chapter4 pages, the fixture line, the full sweep):

```
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 $PY scripts/hf/verify_ppocr_rec_parity.py \
  --checkpoint $MAIN/ppocr-syriac.safetensors \
  --onnx $ART/model.onnx \
  --pages-dir $MAIN/data/dataset/chapter4/pages \
  --xml-dir $MAIN/data/dataset/chapter4/xml \
  --lines $MAIN/tests/fixtures/manuscripts/syriac/transcribe_line.jpg \
  --report-json $ART/parity-report.json
```

with `PY=_ppocr-rec-env/venv/bin/python`,
`ART=_ppocr-rec-env/artifacts/syriac-ppocr-v1`, `MAIN` the main checkout
(read-only inputs). `PYTHONPATH=.` is needed because the scripts import the
worktree's `nomikos_inference` package, same as the house scripts.

## Parity inputs

Real lines: all lines of all 18 Chapter4 pages (`data/dataset/chapter4`),
parsed with kraken's `XMLPage`, cut with kraken's `extract_polygons` on the
baselines path, preprocessed with
`ImageInputTransforms(1, 96, 0, 3, (16, 0))` exactly as
`CTCRecognitionInferenceMixin._recognition_pred` builds it (valid_norm False
on the baselines path, True on the box path; proven to build identical
tensors for 3-channel RGB, so the flag is exactness of construction, not of
pixels), plus
`tests/fixtures/manuscripts/syriac/transcribe_line.jpg` as a plain image.
Two data quirks, both counted in the report: kraken 7.1.1
`parse_page_custom` rejects the platform `custom` attribute format, so
`custom="..."` is stripped from an in-memory copy (geometry and text
untouched); pages store the baseline polyline in the boundary slot, which is
not a polygon, so lines with a degenerate boundary fall back to kraken's own
`BaselineLine.to_bbox` box extraction. Of the 504 Chapter4 lines, 51 had a
real bounding polygon and used polygon extraction; 453 fell back to
bounding-box extraction, because the PAGE-XML stores the baseline polyline in
the boundary (`Coords`) slot. The fallback is kraken's own `to_bbox` (the
`_recognize_box_lines` family), and the report counts both paths per line.
Synthetic lines: seeded uniform noise
in [0, 1] at widths 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 320, 777, 1000,
2000, 3000, 4000 (arithmetic exercise only, no preprocessing claim).

## Parity results (from `parity-report.json`, gate passed, exit 0)

* Real lines: 505 total (504 Chapter4: 51 polygon, 453 box fallback, 0
  skipped; plus the fixture line). Bitwise-equal Torch-vs-ONNX: 0 of 505
  (float kernel noise, expected). Worst max abs logit diff: 1.556e-3 on
  page18#21 (W 3272, T 409), inside the 5e-3 gate; mean of per-line mean
  diffs: 4.17e-05. Worst softmax (over classes, temperature 1.0) max abs
  diff: 2.67e-4, inside the 1e-3 gate. Per-frame argmax agreement: 1.0
  minimum over all real lines. Greedy decodes identical: 505 of 505. ONNX
  shapes equal Torch shapes: all inputs. Torch CER against PAGE-XML text
  (display order both sides, kraken bidi): 0.160.
* Synthetic sweep: noise inputs agree far more tightly than manuscript lines
  (worst max abs diff 2.44e-04 at W 3000) and every width decodes
  identically; full per-width rows are in `parity-report.json` under
  `per_line`.
* Information only: bitwise-equal counts with ORT graph optimizations
  disabled and with `intra_op_num_threads=1` are recorded per line in the
  report (both 0 of 505 on real lines; the gap is deterministic kernel
  arithmetic, not threading or fusion).
* Session options: plain `InferenceSession(path,
  providers=["CPUExecutionProvider"])`, the same construction the Calamari
  adapter uses (`architectures/calamari/adapter.py`, line 80).

## Kraken masking quirk

Kraken's own inference passes `seq_lens` (per-line widths) into the network
(`kraken/models/ctc.py`, both the baseline and box paths), while the exported
graph serves the unmasked result. These are not the same computation:
`_lengths_and_mask` (`kraken/lib/ppocr/network.py`) computes
`floor(W * (w_out / w_in))` in float32, and when `T` is not exactly `W / 8`
the product can land just under `T`, so `out_lens = T - 1` and the last frame
is masked off in attention. Measured: 22 of 505 real lines differ (worst max
abs diff 14.02), and every one of the 22 satisfies `out_lens == T_torch - 1`
(asserted in the parity script; zero violations). 207 of the 4092 widths
5..4096 trigger the shortfall. Only 2 of the 22 change the greedy decode:
page1#12 (masked closer to truth, Levenshtein 13 vs 14) and page10#0
(unmasked closer, 48 vs 49; both far from the truth, that crop reads almost
nothing either way). Side-by-side texts for both are in the parity stdout.
The ONNX artifact serves the unmasked result, so serving can differ from
`kraken ocr` on those widths by design; the quirk is measured and recorded
(`kraken_mask_quirk` in the JSON) and kept out of the gate.

## Minimum working width

Widths 1..4 fail in Torch itself (convolution size guards); width 5 is the
smallest that runs, in both Torch and the ONNX artifact (T=1). The sweep
starts at 8 per the brief; widths 5..7 were probed separately and agree.

## Known limits

* Batch size 1 only. The batch axis is static in the graph, and a padded
  batch gives different logits than single lines in kraken itself (max abs
  diff 16.8 observed), so all parity claims are defined at batch 1.
* The artifact reproduces the Torch graph, not the ground truth: recognition
  quality is whatever the checkpoint holds (see the CER sanity figure above).
* Kraken stays a publish-time dependency: the serving tree never imports it.
