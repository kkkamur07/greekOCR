"""Reference BLLA graph and ONNX exporter.

RETIRED. See ``archive/onnx-runtime/README.md`` and ADR 0004. Originally
``src/model/inference_export/blla/__init__.py``.

Imports throughout this archive are left as they were when the code ran.
``archive/onnx-runtime`` is not an importable package (the directory name has a
hyphen) and nothing here is on the import path; reviving it means restoring the
files to a package location, not fixing these lines in place.
"""

from src.model.inference_export.blla.export import export_blla_onnx

__all__ = ["export_blla_onnx"]
