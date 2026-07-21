"""Export a converted Calamari checkpoint to a self-contained ONNX artifact.

This is development/publishing tooling.  The production inference package
loads the resulting ONNX artifact through ONNX Runtime and does not need this
Torch graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model.inference_export.calamari import export_calamari_onnx


def convert(source: Path, destination: Path) -> None:
    export_calamari_onnx(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="converted Calamari .pt checkpoint")
    parser.add_argument("destination", type=Path, help="self-contained .onnx output")
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
