"""PP-OCR recognition architecture: ONNX Runtime adapter and line preprocessing."""

from nomikos_inference.architectures.ppocr_rec.adapter import (
    PPOCRRecUnavailableError,
    TranscribeLineFailure,
    run_ppocr_rec_transcribe,
    run_ppocr_rec_transcribe_many,
)
from nomikos_inference.architectures.ppocr_rec.preprocessing import (
    preprocess_line_image_bytes_to_ppocr_rec_tensor,
)

__all__ = [
    "PPOCRRecUnavailableError",
    "TranscribeLineFailure",
    "preprocess_line_image_bytes_to_ppocr_rec_tensor",
    "run_ppocr_rec_transcribe",
    "run_ppocr_rec_transcribe_many",
]
