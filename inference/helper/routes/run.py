"""Synchronous inference run for the Inference helper (no service secret)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from inference.admission import CLIENT_INPUT_ERROR, validate_image_bytes
from inference.contracts.run import InferenceRunRequest, InferenceRunResponse
from inference.helper.settings import get_helper_settings
from inference.jobs.runner import run_model

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
            onnx_only=True,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown registry model or tag",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights are not available locally",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=CLIENT_INPUT_ERROR,
        ) from exc
    except RuntimeError as exc:
        # BLLAUnavailableError, CalamariUnavailableError, and the ONNX runtime
        # errors all subclass RuntimeError: the artifact or runtime is broken,
        # not the client request.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference runtime is unavailable for this model",
        ) from exc

    return InferenceRunResponse(task=body.task, output=output)
