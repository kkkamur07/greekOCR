#!/usr/bin/env python3
"""Measure Torch-vs-ONNX parity for one Calamari model before it is published.

ADR 0006 makes ``best.onnx`` the thing a researcher runs and the Torch graph
only the export-time oracle, which means the export is only as trustworthy as
the last time somebody compared the two. ADR 0006 exists *because* a published
BLLA artifact had drifted from its exporter by 1.5e-01 and nothing measured it,
so this script is the measurement, run against the artifact that is about to
ship rather than against a fixture.

What it reports, per line:

* max and mean absolute logits difference (the numeric claim),
* per-timestep argmax agreement (the CTC decision claim: two graphs can differ
  numerically and still pick the same label at every frame),
* whether the greedy-decoded strings are byte-identical (the claim a reader
  cares about, and the only one where "close" is not a pass).

Inputs come from the repo's real preprocessing,
``preprocess_line_image_bytes_to_calamari_tensor``, so the comparison runs on
the tensor the runtime would actually build. Point ``--lines`` at a directory of
real line crops. ``--synthetic-widths`` is the fallback for a script with no
crops on hand, and the report labels it as synthetic: uniform noise exercises
the graph's arithmetic but says nothing about the inputs the model was trained
for, so a decoded-string match on noise is much weaker evidence than one on a
manuscript line.

Example::

    PYTHONPATH=. python scripts/hf/verify_calamari_parity.py \\
        --checkpoint ~/.cache/huggingface/hub/.../best.pt \\
        --onnx var/export/greek/best.onnx \\
        --lines var/lines/greek
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Extensions the line-crop directory is scanned for. Anything else in that
# directory (a labels file, a stray .DS_Store) is skipped rather than fed to
# the decoder as an unreadable image.
LINE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")

# Widths used when no real crops are available. They straddle the two maxpool
# strides so both the odd and the even "same" padding branch is exercised, and
# they run long enough to reach the sequence lengths a real manuscript line
# produces after scaling to the model's line height.
DEFAULT_SYNTHETIC_WIDTHS = (7, 8, 17, 18, 64, 129, 256, 513, 1024)


@dataclass(frozen=True)
class LineParity:
    """One line's worth of agreement between the two graphs."""

    name: str
    timesteps: int
    max_abs_diff: float
    mean_abs_diff: float
    argmax_agreement: float
    torch_text: str
    onnx_text: str

    @property
    def text_identical(self) -> bool:
        return self.torch_text == self.onnx_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Reference best.pt")
    parser.add_argument("--onnx", type=Path, required=True, help="Exported best.onnx")
    parser.add_argument(
        "--lines",
        type=Path,
        help="Directory of real line-crop images to feed both graphs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Maximum number of line crops to run (default: 64)",
    )
    parser.add_argument(
        "--synthetic-widths",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Run synthetic uniform-noise inputs at these pixel widths. "
            f"With no values, uses {DEFAULT_SYNTHETIC_WIDTHS}."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for synthetic inputs")
    parser.add_argument("--report-json", type=Path, help="Write the full report here as JSON")
    return parser.parse_args()


def _softmax(logits: np.ndarray) -> np.ndarray:
    # The same shift-then-normalize the adapter does, so the decoded strings
    # this script compares are produced exactly the way the runtime produces
    # them rather than by a second, subtly different softmax.
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def _line_inputs(directory: Path, *, line_height: int, limit: int) -> list[tuple[str, np.ndarray]]:
    from nomikos_inference.architectures.calamari.preprocessing import (
        preprocess_line_image_bytes_to_calamari_tensor,
    )

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in LINE_IMAGE_SUFFIXES
    )
    inputs: list[tuple[str, np.ndarray]] = []
    for path in paths[:limit]:
        tensor = preprocess_line_image_bytes_to_calamari_tensor(
            path.read_bytes(), line_height=line_height
        ).astype(np.float32, copy=False)
        inputs.append((path.name, tensor))
    return inputs


def _synthetic_inputs(
    widths: tuple[int, ...], *, line_height: int, seed: int
) -> list[tuple[str, np.ndarray]]:
    from nomikos_inference.architectures.calamari.preprocessing import (
        preprocess_line_array_to_calamari_tensor,
    )

    rng = np.random.default_rng(seed)
    inputs: list[tuple[str, np.ndarray]] = []
    for width in widths:
        # Built as a grayscale page-space array and pushed through the same
        # preprocessing as a real crop, so the synthetic path differs from the
        # real one only in where the pixels came from.
        raw = rng.integers(0, 256, size=(line_height, width), dtype=np.uint8)
        tensor = preprocess_line_array_to_calamari_tensor(raw, line_height=line_height).astype(
            np.float32, copy=False
        )
        inputs.append((f"synthetic-w{width}", tensor))
    return inputs


def compare(
    *,
    checkpoint_path: Path,
    onnx_path: Path,
    inputs: list[tuple[str, np.ndarray]],
) -> list[LineParity]:
    """Run both graphs over the same tensors and return the per-line agreement."""
    import onnxruntime as ort
    import torch

    from nomikos_inference.architectures.calamari.adapter import _decode_greedy
    from src.model.inference_export.calamari import load_calamari_checkpoint

    model, metadata = load_calamari_checkpoint(checkpoint_path)
    model.eval()
    charset = list(metadata.charset)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    results: list[LineParity] = []
    for name, image in inputs:
        lengths = np.asarray([image.shape[1]], dtype=np.int64)
        with torch.inference_mode():
            torch_logits = (
                model(torch.from_numpy(image), image_lengths=torch.from_numpy(lengths))["logits"]
                .numpy()[0]
                .astype(np.float64)
            )
        onnx_logits = np.asarray(
            session.run(["logits", "out_len"], {"image": image, "image_lengths": lengths})[0],
            dtype=np.float64,
        )[0]

        difference = np.abs(torch_logits - onnx_logits)
        torch_labels = np.argmax(torch_logits, axis=-1)
        onnx_labels = np.argmax(onnx_logits, axis=-1)
        torch_text, _ = _decode_greedy(_softmax(torch_logits), charset=charset)
        onnx_text, _ = _decode_greedy(_softmax(onnx_logits), charset=charset)
        results.append(
            LineParity(
                name=name,
                timesteps=int(torch_logits.shape[0]),
                max_abs_diff=float(difference.max()),
                mean_abs_diff=float(difference.mean()),
                argmax_agreement=float(np.mean(torch_labels == onnx_labels)),
                torch_text=torch_text,
                onnx_text=onnx_text,
            )
        )
    return results


def summarize(results: list[LineParity], *, source: str) -> dict[str, object]:
    """Roll the per-line rows up into the numbers a release note quotes."""
    if not results:
        raise ValueError("no inputs were compared")
    mismatched = [row.name for row in results if not row.text_identical]
    return {
        "input_source": source,
        "lines": len(results),
        "timesteps_total": sum(row.timesteps for row in results),
        # The maximum over lines, not the mean of the maxima: a single line that
        # drifts is a defect, and averaging it away is how drift ships.
        "max_abs_diff": max(row.max_abs_diff for row in results),
        "mean_abs_diff": float(
            # Weighted by timesteps so a two-frame line does not count as much
            # as a full manuscript line.
            np.average(
                [row.mean_abs_diff for row in results],
                weights=[row.timesteps for row in results],
            )
        ),
        "min_argmax_agreement": min(row.argmax_agreement for row in results),
        "mean_argmax_agreement": float(
            np.average(
                [row.argmax_agreement for row in results],
                weights=[row.timesteps for row in results],
            )
        ),
        "decoded_identical_lines": len(results) - len(mismatched),
        "decoded_mismatched_lines": mismatched,
        "all_decodes_identical": not mismatched,
    }


def main() -> int:
    args = _parse_args()
    if args.lines is None and args.synthetic_widths is None:
        print("one of --lines or --synthetic-widths is required", file=sys.stderr)
        return 1

    from src.model.inference_export.calamari import load_calamari_checkpoint

    _, metadata = load_calamari_checkpoint(args.checkpoint)

    inputs: list[tuple[str, np.ndarray]] = []
    sources: list[str] = []
    if args.lines is not None:
        if not args.lines.is_dir():
            print(f"no such line directory: {args.lines}", file=sys.stderr)
            return 1
        real = _line_inputs(args.lines, line_height=metadata.line_height, limit=args.limit)
        if not real:
            print(f"no line images found under {args.lines}", file=sys.stderr)
            return 1
        inputs += real
        sources.append(f"real line crops from {args.lines}")
    if args.synthetic_widths is not None:
        widths = tuple(args.synthetic_widths) or DEFAULT_SYNTHETIC_WIDTHS
        inputs += _synthetic_inputs(widths, line_height=metadata.line_height, seed=args.seed)
        sources.append(f"synthetic uniform noise at widths {list(widths)}")

    results = compare(checkpoint_path=args.checkpoint, onnx_path=args.onnx, inputs=inputs)
    summary = summarize(results, source="; ".join(sources))
    summary["checkpoint"] = str(args.checkpoint)
    summary["onnx"] = str(args.onnx)
    summary["classes"] = metadata.classes
    summary["lstm_layers"] = metadata.lstm_layers

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.report_json is not None:
        payload = dict(summary)
        payload["per_line"] = [
            {
                "name": row.name,
                "timesteps": row.timesteps,
                "max_abs_diff": row.max_abs_diff,
                "mean_abs_diff": row.mean_abs_diff,
                "argmax_agreement": row.argmax_agreement,
                "text_identical": row.text_identical,
                "torch_text": row.torch_text,
                "onnx_text": row.onnx_text,
            }
            for row in results
        ]
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    # A decoded-string mismatch is the failure ADR 0006 was written about, so
    # it must be an exit status, not a line in a report nobody reads.
    return 0 if summary["all_decodes_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
