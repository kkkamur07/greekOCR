"""Calamari transcribe runner — optional dependency, not used in CI mock mode."""

from __future__ import annotations

from ml.contracts.transcribe import TranscribeRunResponse
from ml.registry import RegistryModelEntry, RegistryVersionEntry
from ml.weights import resolve_weights_source


class CalamariTranscribeRunner:
    def __init__(self, entry: RegistryModelEntry, *, version: RegistryVersionEntry) -> None:
        self._entry = entry
        self._version = version
        self._weights_path = resolve_weights_source(version.weights_source)

    def transcribe(self, image_bytes: bytes, *, params: dict) -> TranscribeRunResponse:
        del image_bytes, params, self._weights_path
        try:
            import calamari_ocr  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "calamari-ocr is not installed; set ML_FORCE_MOCK_RUNNER=1 for tests"
            ) from exc
        raise NotImplementedError(
            "Calamari sync runner wiring is deferred; use ML_FORCE_MOCK_RUNNER=1"
        )
