"""Types shared by the BLLA decoder stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedBLLALine:
    """A decoded line before conversion to the segment contract."""

    baseline: list[list[float]]
    polygon: list[list[float]]
