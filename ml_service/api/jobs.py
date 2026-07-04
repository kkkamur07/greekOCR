"""Async ML job submit routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ml_service.contracts.jobs import JobSubmitRequest, JobSubmitResponse
from ml_service.infrastructure.job_repository import create_job
from ml_service.infrastructure.settings import get_ml_settings
from ml_service.registry import get_model_entry, load_registry

router = APIRouter(prefix="/ml/v1", tags=["jobs"])


@router.post("/jobs", response_model=JobSubmitResponse, status_code=201)
def submit_job(body: JobSubmitRequest) -> JobSubmitResponse:
    settings = get_ml_settings()
    registry = load_registry(settings.ml_registry_path)

    try:
        entry = get_model_entry(registry, body.registry_model_id, body.registry_tag)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if entry.task != body.task:
        raise HTTPException(
            status_code=400,
            detail=(
                f"task {body.task.value!r} does not match registry model "
                f"task {entry.task.value!r}"
            ),
        )

    job = create_job(body)
    return JobSubmitResponse(ml_job_id=job.id)
