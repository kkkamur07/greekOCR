"""Auth request/response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.users.application.password import password_bytes


def _validate_bcrypt_password(value: str) -> str:
    password_bytes(value)
    return value


def _normalize_email(value: str) -> str:
    # Canonicalise to lowercase so ``victim@x.com`` and ``Victim@X.com`` are one
    # account. Postgres ``=`` on the plain email column is case-sensitive, so
    # without this the uniqueness check and its index treat case variants as
    # distinct and let a caller register the same address twice.
    return value.strip().lower()


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return _normalize_email(value)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        return _validate_bcrypt_password(value)


class LoginRequest(BaseModel):
    email: EmailStr
    # Login accepts legacy short passwords; registration enforces the current minimum.
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return _normalize_email(value)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        return _validate_bcrypt_password(value)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - the OAuth2 scheme name, not a token
    #: The browser session's CSRF token, delivered here *as well as* in the
    #: readable ``greekocr-csrf`` cookie.
    #:
    #: The cookie is set on the API host with ``Domain=.nomikos.app`` so that
    #: script on ``app.nomikos.app`` can read it back and echo it in
    #: ``X-CSRF-Token``. That sibling-subdomain read is the fragile step -
    #: Safari's tracking prevention is reported to interfere with it - and when
    #: it fails the client has no way to build the header at all, so every
    #: ``POST /auth/refresh`` answers 403. Handing the same value back in the
    #: body gives the client a second, script-visible channel that does not
    #: depend on cookie policy. The cookie is still set exactly as before.
    #:
    #: Optional in the schema, always populated by the server. The frontend
    #: deploys separately from the API, so a client built against this contract
    #: can be talking to an instance that predates it; the field is typed the
    #: way a client has to treat it rather than the way the server writes it.
    #: ``test_token_response_always_carries_the_csrf_token`` pins the server
    #: side.
    csrf_token: str | None = None


class UserResponse(BaseModel):
    id: UUID
    email: str
    username: str
    # "Use my computer when it is available." Read here so a client that has the
    # account already does not need a second round trip to render the setting;
    # written through ``PUT /account/execution-target``.
    prefer_local_inference: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}
