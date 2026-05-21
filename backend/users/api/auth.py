"""Auth routes: register, login, me."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.users.api.dependencies import get_current_user
from backend.users.api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from backend.users.application.auth_service import AuthService
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(tags=["auth"])
_me_router = APIRouter(tags=["auth"])
_auth = AuthService()


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenResponse:
    _, token = await _auth.register(
        db,
        email=body.email,
        username=body.username,
        password=body.password,
    )
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenResponse:
    try:
        _, token = await _auth.login(db, email=body.email, password=body.password)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        ) from None
    return TokenResponse(access_token=token)


@_me_router.get("/me", response_model=UserResponse)
async def me(current_user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse.model_validate(current_user)
