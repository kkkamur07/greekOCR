"""Artifact production: the trained PyTorch graphs and the ONNX exporters.

Not part of the published wheel and not importable by anything under
``inference/``. ADR 0006 makes ONNX Runtime the inference runtime, so PyTorch's
whole role in this repository is *building* the ``.onnx`` a researcher runs -
which is what lives here, together with the graph definitions the exporters
trace.

The split is the point: ``[project].dependencies`` is what reaches a laptop,
and Torch is not in it. Install this tree's requirements with
``uv run --group export``.
"""
