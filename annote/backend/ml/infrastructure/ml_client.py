"""HTTP client for the root ML inference service."""

from __future__ import annotations

from typing import Any

import httpx

from backend.core.settings.ml import get_ml_settings
from ml.contracts.common import MLTask
from ml.contracts.run import MlRunRequest, MlRunResponse
from ml.contracts.transcribe import TranscribeRunResponse


class MlServiceClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_ml_settings()
        self._base_url = (base_url or settings.ml_service_url).rstrip("/")
        self._transport = transport
        self._timeout = timeout

    async def run(self, request: MlRunRequest) -> MlRunResponse:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            transport=self._transport,
            timeout=self._timeout,
        ) as client:
            response = await client.post(
                "/ml/v1/run",
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            return MlRunResponse.model_validate(response.json())

    async def run_transcribe(
        self,
        *,
        registry_model_id: str,
        registry_tag: str,
        image_bytes: bytes,
        params: dict[str, Any] | None = None,
    ) -> TranscribeRunResponse:
        result = await self.run(
            MlRunRequest(
                task=MLTask.transcribe,
                registry_model_id=registry_model_id,
                registry_tag=registry_tag,
                image_bytes=image_bytes,
                params=params or {},
            )
        )
        if not isinstance(result.output, TranscribeRunResponse):
            raise TypeError("ML service returned non-transcribe output for transcribe task")
        return result.output
