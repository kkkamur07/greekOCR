#!/usr/bin/env python3
"""Export a kraken PP-OCRv6 recognition ``.safetensors`` checkpoint to ``model.onnx``.

This is the step between training (which writes the kraken safetensors file)
and publishing. The artifact this prints the provenance of is what the runtime
opens; the checkpoint travels beside it as the export input.

The report is the operator's half of the registry entry:

* ``source_sha256`` identifies the checkpoint the artifact was traced from,
* ``artifact_sha256`` is what goes into ``registry.yaml`` as ``artifact_sha256``
  and is what the runtime verifies before opening the graph,
* ``classes`` / ``variant`` / ``line_height`` describe the model the artifact
  actually contains, read out of the checkpoint rather than restated elsewhere.

Example::

    PYTHONPATH=. python scripts/hf/export_ppocr_rec_onnx.py \\
        --checkpoint ppocr-syriac.safetensors \\
        --destination var/export/syriac-ppocr-v1/model.onnx \\
        --report-json var/export/syriac-ppocr-v1/export-report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Matches the exporter's own default. It is surfaced as a flag because the
# opset is part of what an artifact is, and a re-export at a different opset
# should be a deliberate, recorded choice rather than a silent consequence of
# upgrading Torch.
DEFAULT_OPSET_VERSION = 17


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local kraken PP-OCRv6 recognition checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Output .onnx path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help=f"ONNX opset version (default: {DEFAULT_OPSET_VERSION})",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Also write the printed report to this path as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from nomikos_inference.export.ppocr_rec import export_ppocr_rec_onnx

    report = export_ppocr_rec_onnx(
        args.checkpoint,
        args.destination,
        opset_version=args.opset,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
