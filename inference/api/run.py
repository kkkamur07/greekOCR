"""Synchronous ML run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from inference.contracts.run import InferenceRunRequest, InferenceRunResponse
from inference.jobs.runner import run_model

router = APIRouter(prefix="/inference/v1", tags=["ml"])


@router.post("/run", response_model=InferenceRunResponse)
def run_inference(body: InferenceRunRequest) -> InferenceRunResponse:
    try:
        output = run_model(
            task=body.task,
            registry_model_id=body.registry_model_id,
            registry_tag=body.registry_tag,
            image_bytes=body.image_bytes,
            params=body.params,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return InferenceRunResponse(task=body.task, output=output)
