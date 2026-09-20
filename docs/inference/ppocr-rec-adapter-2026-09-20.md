# PP-OCR recognition serving adapter (2026-09-20)

The inference server runs the `ppocr_rec` recognition model (`syriac-ppocr-v1`,
a kraken PP-OCR export) on the torch-free ONNX Runtime path, registered
everywhere the `calamari` recognition architecture is registered.

## What the adapter does, step by step

For each line crop, in `nomikos_inference/architectures/ppocr_rec/`:

1. **Preprocess** (`preprocessing.py`): the kraken recognition recipe exactly.
   RGB, aspect-preserving resize to line height 96 with `LANCZOS` and kraken's
   truncating width (`ow = int(w * oh / h)`), white padding (`pad_fill`) on the
   left and right (`pad` columns each), scale to [0, 1] by multiplying with
   `float32(1/255)`, invert (`1 - x`), batch to float32 `[1, 3, 96, W]`. The
   multiply matters: torchvision scales int to float with
   `to(dtype).mul_(1.0 / max)`, and multiply rounds differently from divide on
   some inputs (1 ulp), so `x / 255` is not bitwise identical to kraken.
2. **Run** (`adapter.py`): one ONNX session (`CPUExecutionProvider`), input
   name and output names read from the graph metadata.
3. **Decode**: `softmax(logits / temperature)` over classes in NumPy, greedy
   CTC (argmax per frame, drop blank 0, collapse repeats keeping the max frame
   confidence per emission), class ids mapped through `charset`. A grapheme
   spanning several codepoints emits one confidence per codepoint, duplicated,
   which is what kraken's `PytorchCodec.decode` does and what the response
   contract needs (one `CharacterConfidence` per character of the text).
4. **Reorder** to logical reading order (see below) and return
   `TranscribeRunResponse` exactly as the calamari adapter does, including
   per-line isolation (`TranscribeLineFailure`, all-failed batch re-raises).

Metadata is the source of truth: `line_height`, `pad`, `pad_fill`,
`temperature`, `blank_index`, `classes`, `charset`, `input_name`,
`output_names`, `format` all come from the graph. A graph whose `format` is
not `ppocr-rec-onnx-v1` or whose `blank_index` is not 0 is rejected, mirroring
the calamari checks. Unlike Calamari, where the exporter bakes the
temperature into the graph, the PP-OCR graph emits raw logits and serving
applies the division itself, exactly as kraken's `_rec_predict` divides before
its softmax.

## Bidi decision and vendoring note

The network emits characters in display order (left to right on the image).
Kraken converts to logical reading order in
`BaselineOCRRecord.logical_order`, which calls `kraken.lib.bidi.get_display_map`
with `base_dir=None` (auto resolution) and permutes the confidences with the
same map. The adapter does the same call, so Syriac lines come back in reading
order with confidences attached to the right characters.

The server is torch-free and has no bidi dependency, so kraken's pure-Python
UCD 17.0.0 implementation is vendored at
`nomikos_inference/architectures/ppocr_rec/bidi/` (`__init__.py` plus
`_data.py`, Apache 2.0 header kept). The vendored copy is kraken 7.1.1
normalized by ruff (formatting, import sorting, `Optional` spelling,
`lru_cache` to `cache`, two unused imports removed) plus the import path and
one docstring line noting the source; the algorithm is unchanged. Only
`get_display_map` is used. (The one non-ASCII dash in the tree sits in
kraken's generated `_data.py` header, which says do not edit, so it stays as
kraken wrote it.)

## Narrow-line rule

If the preprocessed width (after padding) is below 16 px, the line is extended
on the right with `pad_fill` up to 16 before inversion, and the reason is noted
in a comment at the rule. The graph was verified from width 16 up. Only the
right side grows: the left padding is part of the trained recipe, while the
right edge past the text carries no signal the backbone was checked without.

## Registration list

* `RegistryArchitecture.ppocr_rec = "ppocr_rec"` (`contracts/common.py`).
* Transcribe dispatch in `jobs/runner.py`, sharing the calamari batch loop
  (crop, per-line isolation, all-failed re-raise) with the runner function and
  failure type threaded through; calamari calls are unchanged.
* `find_hub_artifact` prefers `model.onnx` for `ppocr_rec`
  (`hub/artifacts.py`).
* `seed_dev_inference.py` maps `ppocr_rec` to provider label `kraken` and adds
  `syriac-ppocr-v1` to `TRANSCRIBE_MODELS`.
* No `registry.yaml` entry yet: the Hub revision and sha do not exist until
  the upload, which a later brief does.

## End-to-end results

`scripts/hf/verify_ppocr_rec_adapter.py` (kraken venv) cuts every Chapter4
line with kraken's own XML parser and line extractor (51 polygon lines, 453
box fallback lines, 0 skipped, plus the `transcribe_line.jpg` fixture: 505
lines total) and gates on two comparisons:

* preprocessing bitwise identical (`np.array_equal`): **505 of 505**;
* adapter text identical to kraken's own result (Torch model without
  `seq_lens`, kraken greedy decode plus codec, kraken's own
  `BaselineOCRRecord.logical_order` on the same tensor): **505 of 505**.

For information it also records kraken's masked decode (`seq_lens=[W]`),
which differs on **2** lines (`page1#12`, `page10#0`), the known kraken quirk
the export parity report already names (same two lines there). The full
per-line table is in `adapter-artifacts/adapter-parity.json`.

## Exact commands

From the `ppocr-rec-adapter` worktree, with the repo main venv as `PY` and the
kraken venv as `KPY` (the kraken venv additionally received `pydantic` and
`pydantic-settings` via `uv pip install`, purely so the reference check can
import the adapter; nothing in the worktree depends on them):

```bash
PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/inference/unit -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. $KPY scripts/hf/verify_ppocr_rec_adapter.py \
  --checkpoint $MAIN/ppocr-syriac.safetensors --onnx $ONNX \
  --pages-dir $MAIN/data/dataset/chapter4/pages --xml-dir $MAIN/data/dataset/chapter4/xml \
  --lines $MAIN/tests/fixtures/manuscripts/syriac/transcribe_line.jpg \
  --report-json $OUT/adapter-parity.json
$MAIN/.venv/bin/ruff check <touched files>
$MAIN/.venv/bin/ruff format --check <touched files>
```

Ruff check and format are clean on every touched file, vendored bidi
included. See the pane report for the exact file lists.
