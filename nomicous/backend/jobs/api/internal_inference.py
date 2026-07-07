"""Internal inference routes: job completion callbacks from the inference service."""

from fastapi import APIRouter, Depends, Response, status
from inference.contracts.jobs import JobCallbackRequest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.jobs.api.dependencies import require_inference_webhook_secret
from backend.jobs.application.job_callback_service import JobCallbackService
from infrastructure.db import get_db

router = APIRouter(prefix="/internal/inference", tags=["internal-inference"])


def _callback_service(db: AsyncSession = Depends(get_db)) -> JobCallbackService:
    return JobCallbackService(db)


@router.post(
    "/job-complete",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(require_inference_webhook_secret)],
)
async def complete_inference_job(
    body: JobCallbackRequest,
    service: JobCallbackService = Depends(_callback_service),
) -> Response:
    await service.apply_callback(body)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
