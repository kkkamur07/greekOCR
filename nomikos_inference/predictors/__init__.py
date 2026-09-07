"""Training-stack predictors: run a trained checkpoint through its own framework.

These are not the inference runtime and must not be confused with it.
`nomikos_inference/architectures/calamari/` is the runtime adapter: an ONNX
Runtime session, no TensorFlow, no Torch, no `calamari_ocr` import, and it is
what `run_model` calls on a researcher's laptop (ADR 0006). What is here shells
out to `calamari_ocr.scripts.predict` and loads Transformers checkpoints, which
is the training stack, on the training machine, against the artifact that a
later export step turns into the thing the runtime actually runs.

Two adapters named `calamari` therefore sit in this package, and that is
deliberate rather than a duplicate: `predictors.calamari` runs the export
*input*, `architectures.calamari` runs the export *output*, and collapsing them
would put a `calamari_ocr` dependency back on the inference path.

Placed under `nomikos_inference/` so that this repository has one import root
rather than two, and excluded from both build targets in `pyproject.toml` for
the same reason `[project].dependencies` has no `transformers` in it: nothing
that trains a model is part of what runs one.
"""

from nomikos_inference.predictors.calamari import CalamariPredictor
from nomikos_inference.predictors.trocr import TrOCRPredictor

__all__ = ["CalamariPredictor", "TrOCRPredictor"]
