"""Export the inference-owned BLLA safetensors graph to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model.inference_export.blla import export_blla_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="native BLLA .safetensors path")
    parser.add_argument("destination", type=Path, help="ONNX output path")
    parser.add_argument(
        "--example-width",
        type=int,
        default=64,
        help="width used only while tracing the dynamic-width graph",
    )
    args = parser.parse_args()
    export_blla_onnx(
        args.source,
        args.destination,
        example_width=args.example_width,
    )


if __name__ == "__main__":
    main()
