"""Synchronous ML run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ml.architectures.mock import mock_segment
from ml.contracts.common import MLTask
from ml.contracts.run import MlRunRequest, MlRunResponse
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

    return MlRunResponse(task=body.task, output=mock_segment(body.image_bytes))
