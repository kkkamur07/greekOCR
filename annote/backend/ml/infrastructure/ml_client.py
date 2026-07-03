"""HTTP client for the root ML inference service."""

from __future__ import annotations

from typing import Any

import httpx

from backend.core.settings.ml import get_ml_settings
from backend.document.infrastructure.orm_models import LineGeometryKind
from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult
from ml.contracts.common import MLTask
from ml.contracts.run import MlRunRequest, MlRunResponse
from ml.contracts.segment import SegmentRunResponse


class MlServiceClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_ml_settings()
        self._base_url = (base_url or settings.ml_service_url).rstrip("/")
        self._transport = transport
        self._timeout = timeout

    def run(self, request: MlRunRequest) -> MlRunResponse:
        with httpx.Client(
            base_url=self._base_url,
            transport=self._transport,
            timeout=self._timeout,
        ) as client:
            response = client.post(
                "/ml/v1/run",
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            return MlRunResponse.model_validate(response.json())

    def run_segment(
        self,
        *,
        registry_model_id: str,
        registry_tag: str,
        image_bytes: bytes,
        params: dict[str, Any] | None = None,
    ) -> SegmentRunResponse:
        result = self.run(
            MlRunRequest(
                task=MLTask.segment,
                registry_model_id=registry_model_id,
                registry_tag=registry_tag,
                image_bytes=image_bytes,
                params=params or {},
            )
        )
        if not isinstance(result.output, SegmentRunResponse):
            raise TypeError("ML service returned non-segment output for segment task")
        return result.output

    @staticmethod
    def to_canonical_segment(output: SegmentRunResponse) -> CanonicalSegmentResult:
        return CanonicalSegmentResult(
            blocks=[
                CanonicalBlock(
                    external_id=block.external_id,
                    order=block.order,
                    box=block.box,
                )
                for block in output.blocks
            ],
            lines=[
                CanonicalLine(
                    external_id=line.external_id,
                    order=line.order,
                    block_external_id=line.block_external_id,
                    baseline=line.baseline,
                    mask=line.mask,
                    kind=LineGeometryKind(line.kind.value),
                    points=line.points,
                    kraken_ceiling=line.kraken_ceiling,
                    source_metadata=line.source_metadata,
                )
                for line in output.lines
            ],
        )
