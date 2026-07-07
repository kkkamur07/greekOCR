"""HTTP client for the root inference service."""

from __future__ import annotations

from typing import Any

import httpx
from inference.contracts.jobs import JobSubmitRequest, JobSubmitResponse
from inference.contracts.segment import SegmentRunResponse

from backend.core.settings.ml import get_inference_settings
from backend.document.infrastructure.orm_models import LineGeometryKind
from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult


class InferenceClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_inference_settings()
        self._base_url = (base_url or settings.inference_url).rstrip("/")
        self._transport = transport
        self._timeout = timeout

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(
            base_url=self._base_url,
            transport=self._transport,
            timeout=self._timeout,
        ) as client:
            response = client.post(path, json=body)
            response.raise_for_status()
            return response.json()

    def submit_job(self, request: JobSubmitRequest) -> JobSubmitResponse:
        return JobSubmitResponse.model_validate(
            self._post_json("/inference/v1/jobs", request.model_dump(mode="json"))
        )

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
