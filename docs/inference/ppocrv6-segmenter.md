# PP-OCRv6 medium detection segmenter

## What the model is

Text line detection for manuscript pages with `PaddlePaddle/PP-OCRv6_medium_det`
at Hub revision `8e0f56fb2ef86b461d99cfc7ac5c137738985f61`, served through the
`ppocr-det` runtime adapter in `nomikos_inference/architectures/ppocr_det/` on
ONNX Runtime CPU. The served artifact is the ONNX export
`pp-ocrv6-medium-det.onnx` (SHA-256
`09e4c827c5bb20a0344374bbf8b88d41b7c8bf2be3a0db82ffed3bf090eacfe3`),
whose export and Paddle parity are proved in
[ppocrv6-onnx-parity-2026-09-19.md](ppocrv6-onnx-parity-2026-09-19.md).
The model's Hub page states its licence as Apache-2.0.

## Why it was adopted

Line localisation benchmark (Coptic, 12 pages, 1018 target lines, micro
averaged line localisation, PP box mid axes against annotated polygons, not a
baseline placement score, a development set and not a held out gate):

| Model | P | R | F1 |
|-------|-----|-----|-----|
| BLLA | 61.1 | 84.9 | 71.1 |
| PP-OCRv6 medium cap 960 | 91.8 | 99.1 | 95.3 |
| PP-OCRv6 medium cap 1920 | 90.6 | 99.0 | 94.6 |

Polygon IoU at 0.75: medium 1920 F1 71.6, medium 960 F1 64.1. Greek manual
targets: medium 1920 found 23 of 23 and 13 of 13, medium 960 found 22 of 23
and 11 of 13; the Greek sample is too small for a general claim.

The detector finds nearly every target line where BLLA misses about one in
seven, so it was adopted as a candidate replacement for the default
segmenter, pending the end to end validation below and complete OCR crop
validation.

## How serving works

`run_ppocr_det_segment(image_bytes, model_path, artifact_sha256, params)`
is the entry point production uses. The artifact digest is verified before
the file is opened.

Preprocessing (`preprocessing.py`) reproduces PaddleX 3.7.0 exactly: PIL
decode to RGB, BGR channel order, scale down only when the longer side
exceeds `limit_side_len` with each side rounded to a multiple of 32 (floor
32) via bilinear `cv2.resize`, ImageNet normalisation per stored channel,
CHW float32 batch tensor.

DB postprocess (`postprocessing.py`) mirrors PaddleX 3.7.0 `DBPostProcess`
in quad mode: binarise at `thresh`, `cv2.findContours` over at most
`max_candidates` contours, minimum-area rectangles with a minimum side of 3,
`box_score_fast` gated at `box_thresh`, unclip expansion, a second
minimum-area rectangle with a minimum side of 5, then scaling back to source
coordinates with rounding and clipping. The unclip expansion is PaddleX
3.7.0's own algorithm (OpenCV area and perimeter for the offset distance,
`pyclipper` with `JT_ROUND` for the offsetting), declared in
`[project].dependencies` as `pyclipper>=1.4.0`.

Params and defaults:

| Param | Default | Meaning |
|-------|---------|---------|
| `limit_side_len` | 1920 (320 to 4000) | max side the page is scaled down to |
| `thresh` | 0.2 | probability map binarisation threshold |
| `box_thresh` | 0.45 | minimum mean probability inside a quad |
| `unclip_ratio` | 1.4 | box expansion factor |
| `max_candidates` | 3000 | most contours considered |
| `baseline_fraction` | 0.75 | where the synthetic baseline sits down the quad |
| `reading_direction` | `ltr` | `ltr` or `rtl` column order |

Reading order (`reading_order.py`): quads group transitively into columns by
horizontal overlap, columns run left to right (`ltr`) or right to left
(`rtl`), and inside a column quads group into rows by vertical overlap and
read row by row. A quad spanning two or more columns (a running head or
caption) reads as its own band between the rows above and below it. Narrow
quads (marginal notes, page numbers, initials) attach to the nearest column
without splitting it.

Sessions are created with `ORT_ENABLE_EXTENDED`, sequential execution, 1
inter-op thread and 4 intra-op threads: EXTENDED is about 2x faster than
the ALL default with identical boxes, and 4 threads measured 5.85 s on c13
against 11.0 s at 2 threads and 3.6 s at 8
(docs/inference/ppocrv6-onnx-performance-2026-09-20.md). The intra-op
count comes from `NOMIKOS_PPOCR_DET_THREADS` (integer 1 to 64, default 4);
an invalid value fails the run with a clear error.

## Decisions and their reasons

Boxes, not baselines: PP-OCRv6 emits boxes and the segment contract needs a
baseline per line, so the adapter synthesises one across the quad at
`baseline_fraction` 0.75 of the way from the top edge to the bottom edge,
where kraken baselines sit near the bottom of the letter bodies.

Pyclipper instead of shapely: the unclip call was first written with
shapely `buffer` to avoid a native dependency, but the parity measurement
showed up to 3.162 px corner error with a mean near 0.85 px, so it was
replaced by the exact PaddleX algorithm for full parity (see below).

## End to end validation

`scripts/segmentation/ppocr/verify_adapter.py` feeds each of the 14
reference pages as bytes through the production entry point with
`params=None` and compares the returned quads with the Paddle pipeline
fixtures (`_ppocr-parity/fixtures/`, cap 1920): counts, then greedy
one-to-one matching by quad centre with corner-SET distances. Gate:
identical counts, no unmatched boxes, max corner distance at most 0.5 px.

| page | fixture | adapter | unmatched | max px | mean px |
|------|---------|---------|-----------|--------|---------|
| vat-1r | 78 | 78 | 0 / 0 | 0.000 | 0.000 |
| vat-1v | 71 | 71 | 0 / 0 | 0.000 | 0.000 |
| vat-2r | 69 | 69 | 0 / 0 | 0.000 | 0.000 |
| vat-2v | 64 | 64 | 0 / 0 | 0.000 | 0.000 |
| vat-3r | 69 | 69 | 0 / 0 | 0.000 | 0.000 |
| vat-7v | 67 | 67 | 0 / 0 | 0.000 | 0.000 |
| c10 | 65 | 65 | 0 / 0 | 0.000 | 0.000 |
| c11 | 131 | 131 | 0 / 0 | 0.000 | 0.000 |
| c12 | 130 | 130 | 0 / 0 | 0.000 | 0.000 |
| c13 | 135 | 135 | 0 / 0 | 0.000 | 0.000 |
| c14 | 138 | 138 | 0 / 0 | 0.000 | 0.000 |
| c21 | 130 | 130 | 0 / 0 | 0.000 | 0.000 |
| grec-p1 | 29 | 29 | 0 / 0 | 0.000 | 0.000 |
| grec-p4 | 57 | 57 | 0 / 0 | 0.000 | 0.000 |

The gate passes on all 14 pages: identical counts, nothing unmatched, and
0.000 px corner distance throughout. The PIL decode was checked against a
cv2 decode and is pixel-identical, so the decode contributes nothing. Per-page overlays (`<page>.overlay.jpg` beside `results.json` in
`_ppocr-parity/adapter-e2e/`) show quads in reading order with synthetic
baselines; no fixture box went unmatched on any page.

Reading order spot check on the overlays: two-column spreads read left page
first, each column top to bottom (c13: 1-33, 34-65, then 66-100,
101-102, 103-135; grec-p4: left page 1-26, right page 27-56). Page headers
(shelfmarks, c13 box 1, grec-p1 boxes 1-2, vat-1r boxes 38-39) read before
their column and footers (copyright bar vat-1r 76-78, Gallica bar grec-p1
29 and grec-p4 57) read last. Enlarged initials stay inside their line
boxes (grec-p1 line 3, whose first line is split into boxes 3 and 4 but
reads left to right). Interlinear insertions get their own number in
vertical position (grec-p4 box 49). Watermarks, page labels and marginal
text are returned as ordinary lines interleaved with body text (vat-1r
diagonal library stamps 1-4 and right-margin fragments 57, 59, 62, 64),
which is expected from the known gaps below.

## Publication and registry

The artifact is published at
`hf://nomikos-project/segmentation-ppocrv6-det@stable` (file
`ppocrv6-det.onnx`), pinned in `nomikos_inference/registry.yaml` under
model id `ppocr-segment` by Hub commit
`5091b556c838ce540fde886ead2546fd5feadf62` and artifact SHA-256
`09e4c827c5bb20a0344374bbf8b88d41b7c8bf2be3a0db82ffed3bf090eacfe3`.
The download was verified back from the Hub at the pinned revision before
registering.

How to make it selectable: a row in the platform `inference_models`
catalog table is what exposes a model, inserted by the maintainer; the
page editor has no segment-model picker yet, and the segment endpoint
already accepts `model_id`. Nothing is deployed and this model is not the
default segmenter.

## Known gaps

* No region or column blocks: every line hangs under one full-page block.
* Enlarged initials can be separate boxes.
* Neighbouring boxes can overlap.
* Watermarks and page labels and marginal text are returned as ordinary
  lines.
* The model is not the default segmenter until complete OCR crops are
  validated.
