"""Export a PyTorch Calamari checkpoint as a dynamic-width ONNX model.

The export itself lives in :mod:`nomikos_inference.export.calamari`, which is
the one tracing implementation and metadata writer in the repository: it traces
the length-free LSTM path, so the time axis stays dynamic, and it embeds the
full 12-key metadata set the runtime adapter requires. This module keeps the
trainer-side import path working for the existing callers by delegating to it,
and returns that exporter's checkpoint metadata, which carries the same fields.
Nothing here changes training: no file on the training path imports this module.
"""

from __future__ import annotations

from nomikos_inference.export.calamari import (
    CalamariCheckpointMetadata,
    export_calamari_onnx,
)

__all__ = [
    "CalamariCheckpointMetadata",
    "export_calamari_onnx",
]
