"""Synchronous inference run for the Inference helper (no service secret)."""

from __future__ import annotations

from fastapi import APIRouter

from inference.admission import validate_image_bytes
from inference.contracts.run import InferenceRunRequest, InferenceRunResponse
from inference.helper.settings import get_helper_settings
from inference.jobs.runner import run_model
from inference.run_errors import http_exception_for_run_error

router = APIRouter(prefix="/inference/v1", tags=["ml"])


@router.post("/run", response_model=InferenceRunResponse)
def run_inference(body: InferenceRunRequest) -> InferenceRunResponse:
    try:
        validate_image_bytes(body.image_bytes, get_helper_settings())
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
