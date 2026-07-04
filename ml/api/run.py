"""Synchronous ML run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ml.contracts.common import MLTask
from ml.contracts.run import MlRunRequest, MlRunResponse
from ml.jobs.runner import run_segment
from ml.registry import get_model_entry, load_registry

router = APIRouter(prefix="/ml/v1", tags=["ml"])


@router.post("/run", response_model=MlRunResponse)
def run_ml(body: MlRunRequest) -> MlRunResponse:
    if body.task != MLTask.segment:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"sync run is not supported for task {body.task.value!r}",
        )

    registry = load_registry()
    try:
        entry = get_model_entry(registry, body.registry_model_id, body.registry_tag)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if entry.task != MLTask.segment:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="registry model task does not match segment request",
        )

    try:
        output = run_segment(
            registry_model_id=body.registry_model_id,
            registry_tag=body.registry_tag,
            image_bytes=body.image_bytes,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return MlRunResponse(task=body.task, output=output)
