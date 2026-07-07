"""ML model catalog, scoped bindings, and resolver routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ml.api.schemas import (
    InferenceModelResponse,
    ModelBindingCreateRequest,
    ModelBindingResponse,
    ModelBindingUpdateRequest,
    ResolvedModelBindingResponse,
)
from backend.ml.application.model_service import InferenceModelService
from backend.ml.infrastructure.orm_models import InferenceTask
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(tags=["ml"])
_service = InferenceModelService()


@router.get("/inference/models", response_model=list[InferenceModelResponse])
async def list_inference_models(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InferenceModelResponse]:
    models = await _service.list_models(db)
    return [InferenceModelResponse.model_validate(model) for model in models]


@router.get("/projects/{project_id}/model-bindings", response_model=list[ModelBindingResponse])
async def list_project_model_bindings(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModelBindingResponse]:
    bindings = await _service.list_project_bindings(db, current_user, project_id)
    return [ModelBindingResponse.model_validate(binding) for binding in bindings]


@router.post(
    "/projects/{project_id}/model-bindings",
    response_model=ModelBindingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project_model_binding(
    project_id: UUID,
    body: ModelBindingCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelBindingResponse:
    binding = await _service.create_project_binding(
        db,
        current_user,
        project_id,
        task=body.task,
        model_id=body.model_id,
        overrides=body.overrides,
    )
    return ModelBindingResponse.model_validate(binding)


@router.get(
    "/projects/{project_id}/documents/{document_id}/model-bindings",
    response_model=list[ModelBindingResponse],
)
async def list_document_model_bindings(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModelBindingResponse]:
    bindings = await _service.list_document_bindings(db, current_user, project_id, document_id)
    return [ModelBindingResponse.model_validate(binding) for binding in bindings]


@router.post(
    "/projects/{project_id}/documents/{document_id}/model-bindings",
    response_model=ModelBindingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document_model_binding(
    project_id: UUID,
    document_id: UUID,
    body: ModelBindingCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelBindingResponse:
    binding = await _service.create_document_binding(
        db,
        current_user,
        project_id,
        document_id,
        task=body.task,
        model_id=body.model_id,
        overrides=body.overrides,
    )
    return ModelBindingResponse.model_validate(binding)


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/model-bindings",
    response_model=list[ModelBindingResponse],
)
async def list_part_model_bindings(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModelBindingResponse]:
    bindings = await _service.list_part_bindings(
        db, current_user, project_id, document_id, part_id
    )
    return [ModelBindingResponse.model_validate(binding) for binding in bindings]


@router.post(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/model-bindings",
    response_model=ModelBindingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_part_model_binding(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: ModelBindingCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelBindingResponse:
    binding = await _service.create_part_binding(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        task=body.task,
        model_id=body.model_id,
        overrides=body.overrides,
    )
    return ModelBindingResponse.model_validate(binding)


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/model-bindings/resolve",
    response_model=ResolvedModelBindingResponse,
)
async def resolve_part_model_binding(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    task: InferenceTask,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ResolvedModelBindingResponse:
    resolved = await _service.resolve_for_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        task=task,
    )
    return ResolvedModelBindingResponse(
        binding=ModelBindingResponse.model_validate(resolved.binding),
        model=InferenceModelResponse.model_validate(resolved.model),
        effective_params=resolved.effective_params,
    )


@router.patch(
    "/projects/{project_id}/model-bindings/{binding_id}",
    response_model=ModelBindingResponse,
)
async def update_model_binding(
    project_id: UUID,
    binding_id: UUID,
    body: ModelBindingUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelBindingResponse:
    binding = await _service.update_binding(
        db,
        current_user,
        project_id,
        binding_id,
        model_id=body.model_id,
        overrides=body.overrides,
    )
    return ModelBindingResponse.model_validate(binding)


@router.delete(
    "/projects/{project_id}/model-bindings/{binding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_model_binding(
    project_id: UUID,
    binding_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_binding(db, current_user, project_id, binding_id)
