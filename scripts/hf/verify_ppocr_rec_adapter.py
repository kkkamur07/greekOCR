#!/usr/bin/env python3
"""Prove the PP-OCR recognition serving adapter reproduces kraken's inference.

For every Chapter4 line: cut the line with kraken's own XML parser and line
extractor (bounding-box fallback where the XML has no polygons), encode the
crop as PNG bytes, and compare the adapter against kraken on the same pixels:

(a) the adapter's preprocessed tensor against kraken's
    ``ImageInputTransforms`` output: must be bitwise identical for every line;
(b) the adapter's final text against kraken's own result for the same line
    tensor at batch 1: the kraken Torch model run without ``seq_lens``,
    greedy-decoded with kraken's codec, converted to logical order with
    kraken's own ``BaselineOCRRecord.logical_order``: identical for all lines;
(c) for information, kraken's masked result (``seq_lens=[W]``) and how often
    it differs: a known kraken quirk the export doc explains, not a gate.

Exits 2 unless (a) and (b) hold on every processed line. Run with the kraken
venv (it owns torch and kraken) and ``PYTHONPATH`` pointing at the worktree so
the adapter under test is this checkout::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. $KPY scripts/hf/verify_ppocr_rec_adapter.py \\
        --checkpoint $MAIN/ppocr-syriac.safetensors \\
        --onnx $ONNX --pages-dir $MAIN/data/dataset/chapter4/pages \\
        --xml-dir $MAIN/data/dataset/chapter4/xml \\
        --lines $MAIN/tests/fixtures/manuscripts/syriac/transcribe_line.jpg \\
        --report-json $OUT/adapter-parity.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Reference .safetensors")
    parser.add_argument("--onnx", type=Path, required=True, help="Adapter model.onnx")
    parser.add_argument("--pages-dir", type=Path, help="Directory of page images")
    parser.add_argument("--xml-dir", type=Path, help="Directory of PAGE-XML segmentations")
    parser.add_argument("--lines", type=Path, nargs="*", default=[], help="Extra line images")
    parser.add_argument("--max-lines", type=int, default=None, help="Cap on real lines")
    parser.add_argument("--report-json", type=Path, help="Write the full report here as JSON")
    return parser.parse_args()


def _strip_custom_attributes(xml_text: str) -> str:
    """Drop ``custom="..."`` so kraken's ``parse_page_custom`` accepts the file."""
    return re.sub(r' custom="[^"]*"', "", xml_text)


def _has_real_boundary(line) -> bool:
    """A boundary that is not just the baseline polyline stored twice."""
    if line.boundary is None:
        return False
    return list(map(tuple, line.boundary)) != list(map(tuple, line.baseline))


def _build_real_inputs(pages_dir, xml_dir, line_files, *, height, pad, max_lines):
    """Cut every line with kraken's own extractor; returns crops and counters."""
    from kraken.containers import Segmentation
    from kraken.lib.segmentation import extract_polygons
    from kraken.lib.xml import XMLPage
    from PIL import Image

    inputs = []
    counters = {"pages": 0, "polygon_lines": 0, "box_lines": 0, "skipped_lines": 0}
    budget = max_lines

    def take(line_image, name, truth):
        output = BytesIO()
        line_image.save(output, format="PNG")
        inputs.append(
            {"name": name, "crop": line_image.copy(), "png": output.getvalue(), "truth": truth}
        )

    if xml_dir is not None and pages_dir is not None:
        cleaner = tempfile.TemporaryDirectory(prefix="ppocr-rec-adapter-xml-")
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
                            take(crop, name, xml_line.text)
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
                            take(crop, name, xml_line.text)
                    except (StopIteration, ValueError) as error:
                        counters["skipped_lines"] += 1
                        print(f"cannot cut {name}: {error}", file=sys.stderr)
                        continue
            if budget is not None and len(inputs) >= budget:
                break

    for extra in line_files:
        if budget is not None and len(inputs) >= budget:
            break
        from PIL import Image

        with Image.open(extra) as image:
            take(image.convert("RGB"), Path(extra).name, None)

    return inputs, counters


def _kraken_reference(model, tensor, *, temperature, decoder):
    """Kraken's own decode of one line tensor, without ``seq_lens``.

    Mirrors ``TorchSeqRecognizer.forward`` plus the baseline recognition path:
    temperature softmax, greedy decode, codec to codepoints, then kraken's own
    ``BaselineOCRRecord.logical_order``. The record carries no geometry (only
    the frame cuts kraken itself emits): geometry shapes polygons, never text,
    and the conversion under test is the bidi reorder.
    """
    import torch
    from kraken.containers import BaselineOCRRecord

    with torch.inference_mode():
        logits, olens = model(torch.from_numpy(tensor))
    probs = (logits / temperature).softmax(1)
    outputs = probs.detach().squeeze(2)
    decoded = model.codec.decode(decoder(outputs, olens)[0])
    display = "".join(char for char, _, _, _ in decoded)
    cuts = [[start, end] for _, start, end, _ in decoded]
    confidences = [conf for _, _, _, conf in decoded]
    record = BaselineOCRRecord(
        prediction=display,
        cuts=cuts,
        confidences=confidences,
        line={"type": "baselines", "id": "verify"},
        display_order=True,
    )
    return record.logical_order().prediction


def _kraken_masked_text(model, tensor, width, *, temperature, decoder):
    """Kraken's decode with ``seq_lens=[W]``: the known quirk, info only."""
    import torch

    with torch.inference_mode():
        full, _ = model(torch.from_numpy(tensor), torch.tensor([width]))
    logits = full.numpy()[0, :, 0, :].T
    best = logits.argmax(1)
    labels = {}
    for grapheme, seq in model.codec.c2l.items():
        if len(seq) == 1:
            labels[seq[0]] = grapheme
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
    from kraken.lib.bidi import get_display

    return get_display("".join(parts), None)


def main() -> int:
    args = _parse_args()

    from kraken.configs.base import RecognitionInferenceConfig
    from kraken.lib.ctc_decoder import greedy_decoder
    from kraken.lib.dataset import ImageInputTransforms
    from kraken.models import load_models

    from nomikos_inference.architectures.ppocr_rec.adapter import (
        run_ppocr_rec_transcribe,
    )
    from nomikos_inference.architectures.ppocr_rec.preprocessing import (
        preprocess_line_image_bytes_to_ppocr_rec_tensor,
    )

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

    inputs, counters = _build_real_inputs(
        args.pages_dir,
        args.xml_dir,
        args.lines,
        height=height,
        pad=pad,
        max_lines=args.max_lines,
    )
    if not inputs:
        print("no real line inputs were built", file=sys.stderr)
        return 1

    rows = []
    tensor_identical = 0
    text_identical = 0
    masked_differing = []
    for entry in inputs:
        transforms = ImageInputTransforms(1, height, 0, 3, (pad, 0), False)
        reference = transforms(entry["crop"]).numpy()[None]
        candidate = preprocess_line_image_bytes_to_ppocr_rec_tensor(
            entry["png"], line_height=height, pad=pad, pad_fill=255
        )
        same_tensor = bool(np.array_equal(candidate, reference))
        tensor_identical += int(same_tensor)

        width = int(reference.shape[3])
        kraken_text = _kraken_reference(
            model, reference, temperature=temperature, decoder=greedy_decoder
        )
        adapter_text = run_ppocr_rec_transcribe(entry["png"], checkpoint_path=args.onnx).text
        same_text = kraken_text == adapter_text
        text_identical += int(same_text)

        masked_text = _kraken_masked_text(
            model, reference, width, temperature=temperature, decoder=greedy_decoder
        )
        if masked_text != kraken_text:
            masked_differing.append(entry["name"])

        rows.append(
            {
                "name": entry["name"],
                "width": width,
                "tensor_identical": same_tensor,
                "adapter_text": adapter_text,
                "kraken_text": kraken_text,
                "text_identical": same_text,
                "masked_text": masked_text,
                "truth": entry.get("truth"),
            }
        )

    total = len(rows)
    gate = (
        total > 0
        and counters["skipped_lines"] == 0
        and tensor_identical == total
        and text_identical == total
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.onnx),
        "counters": counters,
        "lines": total,
        "tensor_identical": tensor_identical,
        "text_identical": text_identical,
        "masked_differing": masked_differing,
        "gate": gate,
        "rows": rows,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"lines {total}, preprocessing bitwise identical {tensor_identical}, "
        f"text identical {text_identical}, "
        f"kraken masked-quirk differing {len(masked_differing)}"
    )
    for row in rows[:5]:
        print(f"--- {row['name']}")
        print(f"  adapter: {row['adapter_text']}")
        print(f"  kraken:  {row['kraken_text']}")
        print(f"  truth:   {row['truth']}")
    mismatches = [
        row["name"] for row in rows if not (row["tensor_identical"] and row["text_identical"])
    ]
    if mismatches:
        print(f"mismatched lines ({len(mismatches)}): {mismatches[:20]}", file=sys.stderr)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
