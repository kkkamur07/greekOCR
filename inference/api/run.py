"""Synchronous ML run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from inference.admission import validate_image_bytes
from inference.api.dependencies import require_inference_service_secret
from inference.contracts.run import InferenceRunRequest, InferenceRunResponse
from inference.infrastructure.settings import get_inference_settings
from inference.jobs.runner import run_model
from inference.run_errors import http_exception_for_run_error

router = APIRouter(
    prefix="/inference/v1",
    tags=["ml"],
    dependencies=[Depends(require_inference_service_secret)],
)


@router.post("/run", response_model=InferenceRunResponse)
def run_inference(body: InferenceRunRequest) -> InferenceRunResponse:
    try:
        validate_image_bytes(body.image_bytes, get_inference_settings())
        output = run_model(
            task=body.task,
            registry_model_id=body.registry_model_id,
            registry_tag=body.registry_tag,
            image_bytes=body.image_bytes,
            params=body.params,
        )
    except Exception as exc:
        raise http_exception_for_run_error(exc) from exc

    return InferenceRunResponse(task=body.task, output=output)
