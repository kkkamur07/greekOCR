"""Async inference job submit routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from inference.admission import CLIENT_INPUT_ERROR, validate_image_bytes
from inference.api.dependencies import require_inference_service_secret
from inference.contracts.jobs import JobSubmitRequest, JobSubmitResponse
from inference.infrastructure.job_repository import QueueSaturatedError, create_job
from inference.infrastructure.settings import get_inference_settings
from inference.registry.resolve import resolve_registry_entry

router = APIRouter(
    prefix="/inference/v1",
    tags=["jobs"],
    dependencies=[Depends(require_inference_service_secret)],
)


@router.post("/jobs", response_model=JobSubmitResponse, status_code=201)
def submit_job(body: JobSubmitRequest) -> JobSubmitResponse:
    settings = get_inference_settings()

    try:
        validate_image_bytes(body.image_bytes, settings)
        resolve_registry_entry(
            registry_model_id=body.registry_model_id,
            registry_tag=body.registry_tag,
            task=body.task,
            registry_path=settings.inference_registry_path,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=CLIENT_INPUT_ERROR) from exc

    try:
        job = create_job(body)
    except QueueSaturatedError as exc:
        raise HTTPException(status_code=429, detail="Inference queue is busy; retry later") from exc
    return JobSubmitResponse(inference_job_id=job.id)
