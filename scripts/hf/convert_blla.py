"""Convert Kraken's bundled BLLA Core ML asset to a tensor-only checkpoint.

This is a development/publishing tool. It is intentionally outside the
inference package because the conversion environment may install Kraken and
Core ML Tools; the production inference runtime must not.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from safetensors.torch import save_file

from inference.architectures.blla.blla_model import BLLATorchModel


def convert(source: Path, destination: Path) -> None:
    from kraken.lib import vgsl

    source_model = vgsl.TorchVGSLModel.load_model(source)
    model = BLLATorchModel()
    state_dict = source_model.nn.state_dict()
    model.load_state_dict(state_dict, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        model.state_dict(),
        destination,
        metadata={
            "conversion": "Kraken VGSL state dict mapped to inference BLLATorchModel",
            "format": "blla-pytorch-v1",
            "input_height": str(model.input_height),
            "input_channels": str(model.input_channels),
            "license": "Apache-2.0",
            "output_channels": str(model.output_channels),
            "source_artifact": "kraken/blla.mlmodel",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_version": "kraken==7.0.2",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Kraken blla.mlmodel path")
    parser.add_argument("destination", type=Path, help="native .safetensors output path")
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
