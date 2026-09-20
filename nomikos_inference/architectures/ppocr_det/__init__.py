"""PP-OCRv6 detection segmentation on the ONNX Runtime CPU runtime (ADR 0006)."""

from nomikos_inference.architectures.ppocr_det.ppocr_det import (
    PPOCRDetUnavailableError,
    run_ppocr_det_segment,
)

__all__ = [
    "PPOCRDetUnavailableError",
    "run_ppocr_det_segment",
]
