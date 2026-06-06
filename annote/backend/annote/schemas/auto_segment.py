"""Auto-segmentation request schema."""

from pydantic import BaseModel, Field


class AutoSegmentRequest(BaseModel):
    """Run Kraken line segmentation on a page image."""

    replace: bool = Field(
        default=True,
        description="Replace existing segments. If false, append Kraken lines after current segments.",
    )
    pair_transcription: bool = Field(
        default=True,
        description="Pair segments to transcription text lines in reading order when available.",
    )
