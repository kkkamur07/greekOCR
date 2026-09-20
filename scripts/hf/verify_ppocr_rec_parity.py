#!/usr/bin/env python3
"""Prove a PP-OCRv6 recognition ONNX artifact reproduces its kraken Torch model.

Both graphs run at batch size 1 on the same preprocessed tensors: real Syriac
line images cut from PAGE-XML pages with kraken's own parser, line extractor
and input transforms, plus a synthetic width sweep of seeded uniform noise.
The gate is exact agreement where it matters and a tight numeric bound where
it does not; anything else exits 2.

Inputs come from ``kraken.models.ctc`` territory on purpose. Each PAGE-XML is
parsed with kraken's ``XMLPage`` and each line is cut with kraken's
``extract_polygons`` on the ``baselines`` path, then run through
``ImageInputTransforms(1, height, 0, 3, (pad, 0))`` exactly as
``CTCRecognitionInferenceMixin._recognition_pred`` builds it. Two data quirks
are handled openly rather than worked around silently (both are counted in the
report):

* kraken 7.1.1 ``parse_page_custom`` rejects the platform's ``custom``
  attribute format (``source:kraken; kind:polygon`` has no ``tag{...}`` chunk),
  so ``custom="..."`` attributes are stripped from a copy held in memory.
  Geometry and text are untouched.
* Some pages store the baseline polyline in the boundary slot, which is not a
  polygon. Lines with a degenerate boundary fall back to kraken's own
  ``BaselineLine.to_bbox`` box extraction (the ``_recognize_box_lines`` family).
  Lines are never invented: a line neither path can cut is skipped and counted.

Example::

    PYTHONPATH=. python scripts/hf/verify_ppocr_rec_parity.py \\
        --checkpoint ppocr-syriac.safetensors \\
        --onnx var/export/syriac-ppocr-v1/model.onnx \\
        --pages-dir data/dataset/chapter4/pages \\
        --xml-dir data/dataset/chapter4/xml \\
        --lines tests/fixtures/manuscripts/syriac/transcribe_line.jpg \\
        --report-json var/export/syriac-ppocr-v1/parity-report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

# Uniform-noise widths. They straddle the backbone's stride boundaries (odd
# widths exercise the SAME-padding branches) and run from width 8 to a
# panorama wider than any manuscript line.
DEFAULT_SYNTHETIC_WIDTHS = (8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 320, 777, 1000, 2000, 3000, 4000)

# Line-image file suffixes accepted for --lines and page directories.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")

# Numeric gates. Typical agreement is ~1e-6 relative on logits of magnitude
# tens; the bounds sit an order of magnitude above the worst observed values
# (logits 1.56e-3, softmax 2.67e-4) so a real regression trips them while
# float kernel noise does not. The decisions have their own 100 percent gates.
MAX_ABS_LOGIT_DIFF = 5e-3
MAX_ABS_SOFTMAX_DIFF = 1e-3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Reference .safetensors")
    parser.add_argument("--onnx", type=Path, required=True, help="Exported model.onnx")
    parser.add_argument("--pages-dir", type=Path, help="Directory of page images")
    parser.add_argument("--xml-dir", type=Path, help="Directory of PAGE-XML segmentations")
    parser.add_argument(
        "--lines",
        type=Path,
        nargs="*",
        default=[],
        help="Extra line image files, used as plain images",
    )
    parser.add_argument(
        "--synthetic-widths",
        type=int,
        nargs="*",
        default=None,
        help=f"Noise sweep widths (default: {list(DEFAULT_SYNTHETIC_WIDTHS)})",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Cap on real lines (XML plus --lines), for quick iterations",
    )
    parser.add_argument("--report-json", type=Path, help="Write the full report here as JSON")
    return parser.parse_args()


def _levenshtein(a: str, b: str) -> int:
    """Plain codepoint edit distance; the CER sanity figure needs nothing more."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _strip_custom_attributes(xml_text: str) -> str:
    """Drop ``custom="..."`` so kraken's ``parse_page_custom`` accepts the file."""
    return re.sub(r' custom="[^"]*"', "", xml_text)


def _has_real_boundary(line) -> bool:
    """A boundary that is not just the baseline polyline stored twice."""
    if line.boundary is None:
        return False
    return list(map(tuple, line.boundary)) != list(map(tuple, line.baseline))


def _build_real_inputs(pages_dir, xml_dir, line_files, *, height, pad, max_lines):
    """Cut every line with kraken's own extractor; returns inputs and counters."""
    from kraken.containers import Segmentation
    from kraken.lib.dataset import ImageInputTransforms
    from kraken.lib.segmentation import extract_polygons
    from kraken.lib.xml import XMLPage
    from PIL import Image

    inputs = []
    counters = {"pages": 0, "polygon_lines": 0, "box_lines": 0, "skipped_lines": 0}
    budget = max_lines

    def take(line_image, name, truth, *, valid_norm):
        # valid_norm mirrors CTCRecognitionInferenceMixin._recognition_pred:
        # False on the baselines path, True on the box path. For 3-channel
        # RGB input both build identical transforms (the flag only selects the
        # center-norm branch for single-channel input); it is threaded through
        # so the construction matches kraken's line for line.
        transforms = ImageInputTransforms(1, height, 0, 3, (pad, 0), valid_norm)
        tensor = transforms(line_image).numpy()[None]
        inputs.append({"name": name, "tensor": tensor, "truth": truth})

    if xml_dir is not None and pages_dir is not None:
        cleaner = tempfile.TemporaryDirectory(prefix="ppocr-rec-xml-")
        for xml_path in sorted(xml_dir.glob("*.xml")):
            page_path = None
            for suffix in IMAGE_SUFFIXES:
                candidate = pages_dir / (xml_path.stem + suffix)
                if candidate.is_file():
                    page_path = candidate
                    break
            if page_path is None:
                print(f"no page image for {xml_path.name}, skipping page", file=sys.stderr)
                continue
            cleaned = Path(cleaner.name) / xml_path.name
            cleaned.write_text(
                _strip_custom_attributes(xml_path.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            page = XMLPage(str(cleaned), linetype="baselines")
            if page.type != "baselines" or not page.lines:
                print(f"{xml_path.name} has no baselines, skipping page", file=sys.stderr)
                continue
            counters["pages"] += 1
            with Image.open(page_path) as image:
                image = image.convert("RGB")
                for number, xml_line in enumerate(page.get_sorted_lines()):
                    if budget is not None and len(inputs) >= budget:
                        break
                    name = f"{xml_path.stem}#{number}"
                    try:
                        if _has_real_boundary(xml_line):
                            single = dataclasses.replace(page.to_container(), lines=[xml_line])
                            crop, _ = next(extract_polygons(image, single))
                            counters["polygon_lines"] += 1
                            take(crop, name, xml_line.text, valid_norm=False)
                        else:
                            box = xml_line.to_bbox(text_direction="horizontal-lr")
                            single = Segmentation(
                                type="bbox",
                                imagename=page.imagename,
                                text_direction="horizontal-lr",
                                script_detection=True,
                                lines=[box],
                                regions={},
                            )
                            crop, _ = next(extract_polygons(image, single))
                            counters["box_lines"] += 1
                            take(crop, name, xml_line.text, valid_norm=True)
                    except (StopIteration, ValueError) as error:
                        counters["skipped_lines"] += 1
                        print(f"cannot cut {name}: {error}", file=sys.stderr)
                        continue
            if budget is not None and len(inputs) >= budget:
                break

    for extra in line_files:
        if budget is not None and len(inputs) >= budget:
            break
        with Image.open(extra) as image:
            take(image.convert("RGB"), Path(extra).name, None, valid_norm=False)

    return inputs, counters


def _greedy_decode(logits, labels):
    """Collapse repeats, drop blank 0, map through the codec (display order)."""
    best = logits.argmax(1)
    parts = []
    previous = -1
    for label in best:
        label = int(label)
        if label == 0:
            previous = 0
            continue
        if label != previous:
            parts.append(labels.get(label, ""))
        previous = label
    return "".join(parts)


def _softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> int:
    args = _parse_args()

    import onnxruntime as ort
    import torch
    from kraken.configs.base import RecognitionInferenceConfig
    from kraken.lib.bidi import get_display
    from kraken.models import load_models

    # Inference defaults, read out of kraken rather than restated: pad 16 is
    # the blank margin on each side of a line (configs/base.py), temperature
    # 1.0 leaves the softmax unchanged.
    inference_config = RecognitionInferenceConfig()
    pad = int(inference_config.padding)
    temperature = float(inference_config.temperature)

    models = load_models(str(args.checkpoint))
    if len(models) != 1:
        print(f"expected one model in {args.checkpoint}", file=sys.stderr)
        return 1
    model = models[0]
    model.eval()
    height = int(model.input[2])
    num_classes = int(model.num_classes)
    labels = {}
    for grapheme, seq in model.codec.c2l.items():
        if len(seq) == 1:
            labels[seq[0]] = grapheme

    real_inputs, counters = _build_real_inputs(
        args.pages_dir,
        args.xml_dir,
        args.lines,
        height=height,
        pad=pad,
        max_lines=args.max_lines,
    )
    if not real_inputs:
        print("no real line inputs were built", file=sys.stderr)
        return 1

    widths = (
        tuple(args.synthetic_widths)
        if args.synthetic_widths is not None
        else DEFAULT_SYNTHETIC_WIDTHS
    )
    rng = np.random.default_rng(0)
    synthetic = [
        {
            "name": f"synthetic-w{width}",
            "tensor": rng.random((1, 3, height, width), dtype=np.float32),
            "truth": None,
        }
        for width in widths
    ]

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    narrow = ort.SessionOptions()
    narrow.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_noopt = ort.InferenceSession(
        str(args.onnx), sess_options=narrow, providers=["CPUExecutionProvider"]
    )
    single = ort.SessionOptions()
    single.intra_op_num_threads = 1
    session_single = ort.InferenceSession(
        str(args.onnx), sess_options=single, providers=["CPUExecutionProvider"]
    )

    rows = []
    mask_worst = 0.0
    mask_nonzero_lines = []
    mask_decode_changed = []
    for position, entry in enumerate(real_inputs + synthetic):
        tensor = entry["tensor"]
        width = int(tensor.shape[3])
        masked_logits = None
        masked_out_lens = None
        with torch.inference_mode():
            torch_logits = model(torch.from_numpy(tensor))[0].numpy()[0, :, 0, :].T
            if position < len(real_inputs):
                masked_full, masked_lens = model(torch.from_numpy(tensor), torch.tensor([width]))
                masked_logits = masked_full.numpy()[0, :, 0, :].T
                if masked_lens is not None:
                    masked_out_lens = int(masked_lens.item())
                mask_diff = float(np.abs(torch_logits - masked_logits).max())
                mask_worst = max(mask_worst, mask_diff)
                if mask_diff != 0.0:
                    mask_nonzero_lines.append(entry["name"])
        onnx_raw = np.asarray(session.run(["logits"], {"image": tensor})[0])
        onnx_logits = onnx_raw[0]
        torch_f64 = torch_logits.astype(np.float64)
        onnx_f64 = onnx_logits.astype(np.float64)
        difference = np.abs(torch_f64 - onnx_f64)
        torch_labels = torch_f64.argmax(axis=-1)
        onnx_labels = onnx_f64.argmax(axis=-1)
        torch_text = _greedy_decode(torch_f64, labels)
        onnx_text = _greedy_decode(onnx_f64, labels)
        masked_text = None
        if masked_logits is not None:
            masked_text = _greedy_decode(masked_logits.astype(np.float64), labels)
            if masked_text != torch_text:
                mask_decode_changed.append(entry["name"])
        torch_softmax = _softmax(torch_f64 / temperature)
        onnx_softmax = _softmax(onnx_f64 / temperature)
        rows.append(
            {
                "name": entry["name"],
                "width": width,
                "torch_shape": [1, int(torch_logits.shape[0]), num_classes],
                "onnx_shape": [int(v) for v in onnx_raw.shape],
                "bitwise_equal": bool(np.array_equal(torch_logits, onnx_logits)),
                "max_abs_diff": float(difference.max()),
                "mean_abs_diff": float(difference.mean()),
                "softmax_max_abs_diff": float(np.abs(torch_softmax - onnx_softmax).max()),
                "argmax_agreement": float(np.mean(torch_labels == onnx_labels)),
                "torch_text": torch_text,
                "onnx_text": onnx_text,
                "masked_text": masked_text,
                "masked_max_abs_diff": (
                    float(np.abs(torch_logits - masked_logits).max())
                    if masked_logits is not None
                    else None
                ),
                "masked_out_lens": masked_out_lens,
                "text_identical": torch_text == onnx_text,
                "truth": entry["truth"],
                "noopt_bitwise_equal": bool(
                    np.array_equal(
                        torch_logits,
                        np.asarray(session_noopt.run(["logits"], {"image": tensor})[0])[0],
                    )
                ),
                "single_thread_bitwise_equal": bool(
                    np.array_equal(
                        torch_logits,
                        np.asarray(session_single.run(["logits"], {"image": tensor})[0])[0],
                    )
                ),
            }
        )

    real_rows = rows[: len(real_inputs)]
    synthetic_rows = rows[len(real_inputs) :]
    real_mismatched = [row["name"] for row in real_rows if not row["text_identical"]]
    real_argmax_bad = [row["name"] for row in real_rows if row["argmax_agreement"] != 1.0]
    shape_bad = [row["name"] for row in rows if row["onnx_shape"] != row["torch_shape"]]
    worst_all = max(row["max_abs_diff"] for row in rows)
    worst_real = max(row["max_abs_diff"] for row in real_rows)
    worst_softmax_all = max(row["softmax_max_abs_diff"] for row in rows)
    worst_softmax_real = max(row["softmax_max_abs_diff"] for row in real_rows)

    # CER sanity: Torch display-order decode against the display-order ground
    # truth (kraken's own bidi maps the logical PAGE-XML text to display
    # order). Information only; it says the model and preprocessing are sane.
    cer_dist = cer_len = 0
    for row in real_rows:
        if row["truth"]:
            hypothesis = row["torch_text"]
            reference = get_display(row["truth"])
            cer_dist += _levenshtein(hypothesis, reference)
            cer_len += len(reference)
    cer = cer_dist / cer_len if cer_len else None

    # The masking quirk is kraken's, not the export's: _lengths_and_mask
    # (kraken/lib/ppocr/network.py) scales widths in float32, so floor can land
    # one frame short and the last frame is masked off. The exported graph
    # serves the unmasked result by design, so this is measured, explained and
    # kept out of the gate. Every nonzero line must satisfy out_lens == T - 1;
    # a line that breaks the explanation is an open question, not a silent row.
    quirk_lines = []
    quirk_violations = []
    for row in real_rows:
        if row["name"] not in mask_nonzero_lines:
            continue
        time_steps = row["torch_shape"][1]
        out_lens = row["masked_out_lens"]
        quirk_lines.append(
            {
                "name": row["name"],
                "W": row["width"],
                "T_torch": time_steps,
                "out_lens": out_lens,
            }
        )
        if out_lens != time_steps - 1:
            quirk_violations.append(row["name"])
    quirk_changed = []
    for name in mask_decode_changed:
        row = next(row for row in real_rows if row["name"] == name)
        reference = get_display(row["truth"]) if row["truth"] else None
        entry = {
            "name": name,
            "unmasked_text": row["torch_text"],
            "masked_text": row["masked_text"],
            "truth_display": reference,
        }
        if reference is not None:
            lev_unmasked = _levenshtein(row["torch_text"], reference)
            lev_masked = _levenshtein(row["masked_text"], reference)
            entry["levenshtein_unmasked"] = lev_unmasked
            entry["levenshtein_masked"] = lev_masked
            entry["closer_to_truth"] = "unmasked" if lev_unmasked <= lev_masked else "masked"
        quirk_changed.append(entry)

    gate = {
        "all_real_decodes_identical": not real_mismatched,
        "all_real_argmax_agree": not real_argmax_bad,
        "max_abs_diff_within_bound": worst_all <= MAX_ABS_LOGIT_DIFF,
        "softmax_within_bound": worst_softmax_all <= MAX_ABS_SOFTMAX_DIFF,
        "all_shapes_match_torch": not shape_bad,
    }
    passed = all(gate.values())

    summary = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.onnx),
        "classes": num_classes,
        "ort_version": ort.__version__,
        "torch_version": torch.__version__,
        "counters": counters,
        "real_lines": len(real_rows),
        "synthetic_lines": len(synthetic_rows),
        "bitwise_equal_real": sum(row["bitwise_equal"] for row in real_rows),
        "bitwise_equal_noopt_real": sum(row["noopt_bitwise_equal"] for row in real_rows),
        "bitwise_equal_single_thread_real": sum(
            row["single_thread_bitwise_equal"] for row in real_rows
        ),
        "worst_max_abs_diff_real": worst_real,
        "worst_max_abs_diff_all": worst_all,
        "worst_softmax_max_abs_diff_real": worst_softmax_real,
        "worst_softmax_max_abs_diff_all": worst_softmax_all,
        "mean_of_mean_abs_diff_real": float(np.mean([row["mean_abs_diff"] for row in real_rows])),
        "min_argmax_agreement_real": min(row["argmax_agreement"] for row in real_rows),
        "decoded_identical_real": len(real_rows) - len(real_mismatched),
        "decoded_mismatched_real": real_mismatched,
        "argmax_mismatched_real": real_argmax_bad,
        "shape_mismatched": shape_bad,
        "kraken_mask_quirk": {
            "max_abs_diff": mask_worst,
            "lines": quirk_lines,
            "out_lens_always_T_minus_1": not quirk_violations,
            "explanation_violations": quirk_violations,
            "decode_changed": quirk_changed,
        },
        "torch_cer_vs_xml": cer,
        "gate": gate,
        "gate_passed": passed,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    shown = 0
    for row in real_rows:
        if shown >= 5 or not row["truth"]:
            continue
        print(f"--- {row['name']}")
        print(f"  torch: {row['torch_text']}")
        print(f"  onnx : {row['onnx_text']}")
        print(f"  truth: {get_display(row['truth'])}")
        shown += 1

    for entry in quirk_changed:
        print(f"--- mask quirk {entry['name']}")
        print(f"  unmasked: {entry['unmasked_text']}")
        print(f"  masked  : {entry['masked_text']}")
        print(f"  truth   : {entry['truth_display']}")
        if "closer_to_truth" in entry:
            print(
                f"  levenshtein unmasked={entry['levenshtein_unmasked']} "
                f"masked={entry['levenshtein_masked']} "
                f"closer={entry['closer_to_truth']}"
            )

    if args.report_json is not None:
        payload = dict(summary)
        payload["per_line"] = rows
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    # A decoded-string mismatch is the failure ADR 0006 was written about, so
    # it must be an exit status, not a line in a report nobody reads.
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
