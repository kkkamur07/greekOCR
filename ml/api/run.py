"""Synchronous ML run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ml.architectures.runner import transcribe_from_registry
from ml.contracts.common import MLTask
from ml.contracts.run import MlRunRequest, MlRunResponse
from ml.registry import get_model_entry, load_registry

router = APIRouter(prefix="/ml/v1", tags=["ml"])


@router.post("/run", response_model=MlRunResponse)
def run_ml(body: MlRunRequest) -> MlRunResponse:
    if body.task == MLTask.transcribe:
        registry = load_registry()
        try:
            entry = get_model_entry(registry, body.registry_model_id, body.registry_tag)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        if entry.task != MLTask.transcribe:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="registry model task does not match transcribe request",
            )
        output = transcribe_from_registry(
            registry_model_id=body.registry_model_id,
            registry_tag=body.registry_tag,
            image_bytes=body.image_bytes,
            params=body.params,
        )
        return MlRunResponse(task=body.task, output=output)

    if body.task == MLTask.segment:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="segment sync run is not implemented yet",
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"unsupported task: {body.task.value}",
    )
