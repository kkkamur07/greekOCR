"""Shared FastAPI security dependencies."""

from fastapi.security import HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False)
