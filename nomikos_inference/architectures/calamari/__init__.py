"""Calamari OCR architecture: ONNX Runtime adapter and line preprocessing.

The Torch graph that used to be re-exported here (``CalamariTorchModel`` and
its config) is not part of the runtime under ADR 0006. It builds the artifact
rather than running it, so it lives in ``src/model/inference_export/calamari/``
and is not in the
published wheel.
"""

from nomikos_inference.architectures.calamari.adapter import (
    CalamariUnavailableError,
    TranscribeLineFailure,
    run_calamari_transcribe,
    run_calamari_transcribe_many,
)
from nomikos_inference.architectures.calamari.preprocessing import (
    preprocess_line_image_to_calamari_tensor,
)

__all__ = [
    "CalamariUnavailableError",
    "TranscribeLineFailure",
    "preprocess_line_image_to_calamari_tensor",
    "run_calamari_transcribe",
    "run_calamari_transcribe_many",
]
