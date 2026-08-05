"""Regressions for the authentication and authorization hardening pass.

Covers the JWT signing-key floor, access-token revocation, registration account
enumeration, the environment gate on the development seed, production docs
exposure, and client attribution for rate limiting.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt import InvalidTokenError
from pydantic import ValidationError
from starlette.requests import Request

from backend.core.exceptions import AccessDeniedError, ConflictError
from backend.core.settings import (
    AppSettings,
    AuthSettings,
    get_app_settings,
    get_auth_settings,
    get_infrastructure_settings,
    get_job_settings,
    get_ml_settings,
    get_storage_settings,
    reset_settings_caches,
)
from backend.core.settings.auth import (
    MIN_JWT_SECRET_BYTES,
    MIN_JWT_SECRET_GUESSES_LOG10,
    secret_guesses_log10,
)
from backend.core.settings.device import get_device_settings
from backend.users.api.dependencies import get_current_user
from backend.users.api.rate_limit import (
    attributable_client_ip,
    auth_rate_limit_keys,
    _real_ip,
)
from backend.users.application.auth_service import REGISTRATION_CONFLICT_MESSAGE, AuthService
from backend.users.application.jwt_tokens import create_access_token, decode_access_token
from tests.fixtures.paths import REPO_ROOT

# A real 32-byte urlsafe token; used wherever a test needs a production-grade key.
STRONG_SECRET = "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV"


def _clear_platform_settings() -> None:
    # This list happened to be complete; the two others like it were not. Sharing
    # one registry-backed reset is what stops that from being luck.
    reset_settings_caches()


@pytest.fixture(autouse=True)
def clear_platform_settings_cache():
    _clear_platform_settings()
    yield
    _clear_platform_settings()


# --- JWT signing key floor ---
# Tests length and entropy gating of JWT_SECRET. Does not test token signing itself.


@pytest.mark.parametrize(
    "secret",
    [
        "short-but-not-a-placeholder",  # 27 bytes
        "a" * (MIN_JWT_SECRET_BYTES - 1),
    ],
)
def test_production_rejects_jwt_secret_below_the_length_floor(secret: str) -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        AuthSettings(ENVIRONMENT="production", JWT_SECRET=secret, _env_file=None)


@pytest.mark.parametrize(
    "secret",
    [
        "a" * 64,
        "abababababababababababababababababababab",
        "secretsecretsecretsecretsecretsecret",
    ],
)
def test_production_rejects_low_entropy_jwt_secret_that_clears_the_length_floor(
    secret: str,
) -> None:
    assert len(secret.encode()) >= MIN_JWT_SECRET_BYTES
    with pytest.raises(ValidationError, match="too guessable"):
        AuthSettings(ENVIRONMENT="production", JWT_SECRET=secret, _env_file=None)


def test_production_accepts_a_generated_secret() -> None:
    settings = AuthSettings(ENVIRONMENT="production", JWT_SECRET=STRONG_SECRET, _env_file=None)

    assert settings.jwt_secret == STRONG_SECRET
    assert secret_guesses_log10(STRONG_SECRET) >= MIN_JWT_SECRET_GUESSES_LOG10


def test_development_keeps_short_secrets_workable() -> None:
    """The floor is a production gate; a local secret must not break `next dev`."""
    settings = AuthSettings(
        ENVIRONMENT="development", JWT_SECRET="local-dev-secret", _env_file=None
    )

    assert settings.jwt_secret == "local-dev-secret"


@pytest.mark.parametrize("environment", ["development", "production"])
def test_placeholder_secrets_are_rejected_in_every_environment(environment: str) -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        AuthSettings(ENVIRONMENT=environment, JWT_SECRET="replace-me", _env_file=None)


def test_shipped_example_secrets_still_fail_validation() -> None:
    """The committed templates must never be a working configuration."""
    compose = (REPO_ROOT / ".env.compose.example").read_text(encoding="utf-8")
    production = (
        REPO_ROOT / "nomicous" / "backend" / "core" / ".env.production.example"
    ).read_text(encoding="utf-8")

    secrets = [
        line.split("=", 1)[1].strip()
        for line in (compose + production).splitlines()
        if line.startswith("JWT_SECRET=")
    ]
    assert secrets, "no JWT_SECRET placeholder found in the shipped templates"
    for secret in secrets:
        with pytest.raises(ValidationError, match="JWT_SECRET"):
            AuthSettings(ENVIRONMENT="production", JWT_SECRET=secret, _env_file=None)


def test_the_estimate_separates_generated_secrets_from_memorable_ones() -> None:
    """The gap the threshold sits in, asserted so a zxcvbn upgrade cannot close it."""
    assert secret_guesses_log10("") == 0.0
    assert secret_guesses_log10("a" * 32) < MIN_JWT_SECRET_GUESSES_LOG10
    # Strong as a human password - zxcvbn scores both a perfect 4/4 - and nowhere
    # near strong enough to sign tokens with. This is why the gate reads
    # `guesses_log10` and not `score`.
    assert secret_guesses_log10("correcthorsebatterystaple1234567") < MIN_JWT_SECRET_GUESSES_LOG10
    assert secret_guesses_log10(secrets.token_hex(16)) > MIN_JWT_SECRET_GUESSES_LOG10


@pytest.mark.parametrize(
    "generate",
    [
        # The message on the failure this gate raises recommends the first one.
        lambda: secrets.token_urlsafe(32),
        lambda: secrets.token_hex(16),
        lambda: secrets.token_hex(32),
    ],
)
def test_production_accepts_every_recommended_way_of_generating_a_secret(generate) -> None:
    """Refusing a correctly generated secret is not a strict gate, it is an outage.

    A hand-rolled character-frequency estimate used to reject `token_hex(16)` on
    2000 draws out of 2000. zxcvbn's worst observed draw over 5000 was 27.45,
    comfortably clear of the 22.0 floor.
    """
    for _ in range(200):
        secret = generate()
        assert (
            AuthSettings(ENVIRONMENT="production", JWT_SECRET=secret, _env_file=None).jwt_secret
            == secret
        )


@pytest.mark.parametrize(
    "secret",
    [
        "a" * 32,
        ("abc" * 11)[:32],
        ("abc" * 22)[:64],
        "password" * 4,
        "0123456789" * 4,  # a distribution-only estimate scored this 133 and passed it
        "0123456789abcdef" * 2,  # a perfectly uniform alphabet, and still a repeat
        "correcthorsebatterystaple1234567",  # zxcvbn score 4/4, still not a signing key
    ],
)
def test_patterned_secrets_of_the_same_length_are_still_rejected(secret: str) -> None:
    assert len(secret) >= 32
    assert secret_guesses_log10(secret) < MIN_JWT_SECRET_GUESSES_LOG10


# --- Credentialed CORS allowlist ---
# Tests the production template's origin list. Does not test CORSMiddleware behaviour.


def test_production_cors_template_excludes_the_marketing_apex() -> None:
    template = (REPO_ROOT / "nomicous" / "backend" / "core" / ".env.production.example").read_text(
        encoding="utf-8"
    )
    line = next(line for line in template.splitlines() if line.startswith("CORS_ORIGINS="))
    origins = line.split("=", 1)[1].split(",")

    assert origins == ["https://app.nomicous.com"]
    assert "https://nomicous.com" not in origins


# --- Access-token revocation ---
# Tests that a token is only accepted while its browser session is live.
# Does not test session rotation or CSRF, which integration tests cover.


class _Result:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _FakeSession:
    """Minimal AsyncSession stand-in that records the statement it was given."""

    def __init__(self, value: object = None) -> None:
        self.value = value
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.value)


def _credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_access_token_carries_its_session_id() -> None:
    settings = AuthSettings(JWT_SECRET=STRONG_SECRET, _env_file=None)
    user_id, session_id = uuid.uuid4(), uuid.uuid4()

    claims = decode_access_token(
        create_access_token(user_id, settings, session_id=session_id), settings
    )

    assert claims.user_id == user_id
    assert claims.session_id == session_id


async def test_token_without_a_session_claim_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    _clear_platform_settings()
    settings = get_auth_settings()
    token = create_access_token(uuid.uuid4(), settings)

    with pytest.raises(HTTPException) as exc:
        await get_current_user(_credentials(token), _FakeSession(object()))

    assert exc.value.status_code == 401


async def test_revoked_session_invalidates_a_still_unexpired_token(monkeypatch) -> None:
    """The whole point: logout must not leave the JWT usable until `exp`."""
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    _clear_platform_settings()
    settings = get_auth_settings()
    token = create_access_token(uuid.uuid4(), settings, session_id=uuid.uuid4())
    # No row comes back once the session is revoked or expired.
    db = _FakeSession(None)

    with pytest.raises(HTTPException) as exc:
        await get_current_user(_credentials(token), db)

    assert exc.value.status_code == 401
    assert decode_access_token(token, settings).session_id is not None


async def test_live_session_still_authenticates_and_filters_on_revocation(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    _clear_platform_settings()
    settings = get_auth_settings()
    user = object()
    db = _FakeSession(user)

    resolved = await get_current_user(
        _credentials(create_access_token(uuid.uuid4(), settings, session_id=uuid.uuid4())), db
    )

    assert resolved is user
    sql = str(db.statements[0]).lower()
    assert "revoked_at is null" in sql
    assert "expires_at >" in sql


def test_tampered_token_is_rejected() -> None:
    settings = AuthSettings(JWT_SECRET=STRONG_SECRET, _env_file=None)
    other = AuthSettings(JWT_SECRET=STRONG_SECRET[::-1], _env_file=None)
    token = create_access_token(uuid.uuid4(), other, session_id=uuid.uuid4())

    with pytest.raises(InvalidTokenError):
        decode_access_token(token, settings)


# --- Registration enumeration ---
# Tests that conflict responses are indistinguishable. Does not test password hashing.


class _StubUserRepository:
    def __init__(self, *, email_taken: bool = False, username_taken: bool = False) -> None:
        self._email_taken = email_taken
        self._username_taken = username_taken

    async def get_by_email(self, session, email):
        return object() if self._email_taken else None

    async def get_by_username(self, session, username):
        return object() if self._username_taken else None


@pytest.mark.parametrize(
    "kwargs",
    [{"email_taken": True}, {"username_taken": True}],
)
async def test_registration_conflicts_are_indistinguishable(kwargs, monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    _clear_platform_settings()
    service = AuthService(
        repository=_StubUserRepository(**kwargs), auth_settings=get_auth_settings()
    )

    with pytest.raises(ConflictError) as exc:
        await service.register(
            None, email="taken@example.com", username="taken", password="password123"
        )

    assert str(exc.value) == REGISTRATION_CONFLICT_MESSAGE


def test_registration_conflict_message_names_neither_field() -> None:
    lowered = REGISTRATION_CONFLICT_MESSAGE.casefold()

    assert "email" not in lowered
    assert "username" not in lowered


# --- Development seed gate ---
# Tests the environment guard on the published dev credentials.


@pytest.mark.parametrize("environment", ["production", "staging", "test"])
async def test_dev_user_seed_refuses_to_run_outside_development(environment, monkeypatch) -> None:
    from backend.dev.bootstrap import ensure_dev_user_exists, reset_dev_user_password

    monkeypatch.setenv("ENVIRONMENT", environment)
    _clear_platform_settings()

    for seed in (ensure_dev_user_exists, reset_dev_user_password):
        with pytest.raises(RuntimeError, match="development"):
            await seed(None)


# --- Production docs exposure ---
# Tests that the schema and interactive docs are not served in production.


def _production_env(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    monkeypatch.setenv("CORS_ORIGINS", "https://app.nomicous.com")
    monkeypatch.setenv("CLOUD_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("JOB_WORKER_ENABLED", "false")
    monkeypatch.setenv("INFERENCE_URL", "https://inference.example.com")
    monkeypatch.setenv("INFERENCE_WEBHOOK_SECRET", "unit-test-webhook-secret")
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "unit-test-service-secret")
    _clear_platform_settings()


def test_production_serves_no_docs_or_openapi_schema(monkeypatch) -> None:
    from backend.core.app import create_app

    _production_env(monkeypatch)
    app = create_app()

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    served = {route.path for route in app.routes}
    assert "/docs" not in served
    assert "/redoc" not in served
    assert "/openapi.json" not in served


def test_development_keeps_the_docs(monkeypatch) -> None:
    from backend.core.app import create_app

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("JWT_SECRET", "local-dev-secret")
    _clear_platform_settings()
    app = create_app()

    assert app.docs_url == "/docs"
    assert app.openapi_url == "/openapi.json"


async def test_root_route_does_not_advertise_docs_in_production(monkeypatch) -> None:
    from backend.core.api.root import welcome

    monkeypatch.setenv("ENVIRONMENT", "production")
    _clear_platform_settings()

    assert (await welcome()).docs_url is None


# --- Client attribution for rate limiting ---
# Tests which bucket keys an auth attempt is charged against.
# Does not exercise the Postgres store, which needs a database.


def _request(
    *,
    peer: str | None,
    forwarded_for: str | None = None,
    body: bytes | None = None,
    content_type: str | None = "application/json",
    path: str = "/auth/login",
):
    headers = [(b"host", b"testserver")]
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    if body is not None and content_type is not None:
        headers.append((b"content-type", content_type.encode()))
    request = Request(
        {
            "type": "http",
            "headers": headers,
            "client": (peer, 12345) if peer else None,
            "scheme": "http",
            "method": "POST",
            "path": path,
            "query_string": b"",
            "server": ("testserver", 80),
        }
    )
    if body is not None:
        request._body = body
    return request


def test_untrusted_peer_is_not_treated_as_a_client_address(monkeypatch) -> None:
    """Vercel's peer is the platform proxy; keying on it makes one global bucket."""
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    monkeypatch.setenv("BEHIND_PROXY", "false")
    _clear_platform_settings()

    assert attributable_client_ip(_request(peer="10.0.0.1")) is None
    # The legacy helper still resolves something, which is exactly why the
    # limiter must not use it for a per-client decision.
    assert _real_ip(_request(peer="10.0.0.1")) == "10.0.0.1"


def test_direct_deployments_still_attribute_the_peer(monkeypatch) -> None:
    monkeypatch.setenv("TRUST_PEER_IP", "true")
    monkeypatch.setenv("BEHIND_PROXY", "false")
    _clear_platform_settings()

    assert attributable_client_ip(_request(peer="203.0.113.7")) == "203.0.113.7"


def test_trusted_proxy_still_wins_over_the_peer(monkeypatch) -> None:
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    monkeypatch.setenv("BEHIND_PROXY", "true")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    _clear_platform_settings()

    request = _request(peer="10.1.2.3", forwarded_for="203.0.113.10, 10.2.3.4")
    assert attributable_client_ip(request) == "203.0.113.10"


def test_spoofed_forwarded_header_is_ignored_from_an_untrusted_peer(monkeypatch) -> None:
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    monkeypatch.setenv("BEHIND_PROXY", "true")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    _clear_platform_settings()

    request = _request(peer="198.51.100.9", forwarded_for="203.0.113.10")
    assert attributable_client_ip(request) is None


async def test_auth_throttle_falls_back_to_a_per_account_bucket(monkeypatch) -> None:
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    monkeypatch.setenv("BEHIND_PROXY", "false")
    _clear_platform_settings()
    digest = hashlib.sha256(b"victim@example.com").hexdigest()

    keys = await auth_rate_limit_keys(
        _request(peer="10.0.0.1", body=b'{"email":"Victim@Example.com ","password":"x"}')
    )

    assert keys == [f"account:{digest}:/auth/login"]


async def test_auth_throttle_charges_both_dimensions_when_the_ip_is_usable(monkeypatch) -> None:
    monkeypatch.setenv("TRUST_PEER_IP", "true")
    monkeypatch.setenv("BEHIND_PROXY", "false")
    _clear_platform_settings()
    digest = hashlib.sha256(b"victim@example.com").hexdigest()

    keys = await auth_rate_limit_keys(
        _request(peer="203.0.113.7", body=b'{"email":"victim@example.com","password":"x"}')
    )

    assert keys == ["ip:203.0.113.7:/auth/login", f"account:{digest}:/auth/login"]


async def test_account_bucket_never_stores_the_plain_email(monkeypatch) -> None:
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    _clear_platform_settings()

    keys = await auth_rate_limit_keys(
        _request(peer="10.0.0.1", body=b'{"email":"victim@example.com","password":"x"}')
    )

    assert keys
    assert all("victim@example.com" not in key for key in keys)


@pytest.mark.parametrize(
    "content_type",
    [
        "Application/JSON",
        "APPLICATION/JSON",
        "application/json; charset=UTF-8",
        "Application/JSON;charset=utf-8",
        "application/vnd.api+json",
        None,  # no Content-Type header at all - FastAPI still parses the body
    ],
)
async def test_account_bucket_ignores_how_the_body_is_labelled(monkeypatch, content_type) -> None:
    """Regression: a case-sensitive media-type gate switched off the account key.

    Media types are case-insensitive (RFC 9110), FastAPI parses any
    `application/*+json` subtype, and it parses a body with no declared type at
    all. On Vercel there is no IP key to fall back on, so `Application/JSON` was
    unmetered password guessing. Every labelling below must throttle identically.
    """
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    monkeypatch.setenv("BEHIND_PROXY", "false")
    _clear_platform_settings()
    digest = hashlib.sha256(b"victim@example.com").hexdigest()
    body = b'{"email":"victim@example.com","password":"x"}'

    lowercase = await auth_rate_limit_keys(_request(peer="10.0.0.1", body=body))
    relabelled = await auth_rate_limit_keys(
        _request(peer="10.0.0.1", body=body, content_type=content_type)
    )

    assert lowercase == [f"account:{digest}:/auth/login"]
    assert relabelled == lowercase


async def test_body_too_large_to_attribute_is_refused_rather_than_waved_through(
    monkeypatch,
) -> None:
    """Pydantic ignores unknown keys, so a padded login body still authenticates.

    Skipping identity extraction on it would put the account bucket one `pad`
    field away from being switched off, so the request is rejected instead.
    """
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    _clear_platform_settings()

    oversized = b'{"email":"a@b.com","password":"guess","pad":"' + b"x" * 9000 + b'"}'
    with pytest.raises(HTTPException) as exc:
        await auth_rate_limit_keys(_request(peer="10.0.0.1", body=oversized))

    assert exc.value.status_code == 413


async def test_non_json_and_empty_bodies_produce_no_account_key(monkeypatch) -> None:
    """Neither can reach a password check: the routes only bind pydantic from JSON."""
    monkeypatch.setenv("TRUST_PEER_IP", "false")
    _clear_platform_settings()

    assert await auth_rate_limit_keys(_request(peer="10.0.0.1")) == []
    assert (
        await auth_rate_limit_keys(
            _request(peer="10.0.0.1", body=b"email=victim%40example.com&password=x")
        )
        == []
    )
    assert await auth_rate_limit_keys(_request(peer="10.0.0.1", body=b"[" * 5000)) == []


async def test_unattributable_request_is_charged_not_waved_through(monkeypatch) -> None:
    """The fail-open branch was the third link in the bypass chain.

    A coarse shared bucket is safe *here* only because no real sign-in lands in
    it - a body with no `email` never reaches password verification.
    """
    from backend.users.api import rate_limit

    monkeypatch.setenv("TRUST_PEER_IP", "false")
    monkeypatch.setenv("BEHIND_PROXY", "false")
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    _clear_platform_settings()
    calls: list[tuple[list[str], int]] = []

    async def _record(keys, *, limit, window_seconds, detail):
        calls.append((list(keys), limit))

    monkeypatch.setattr(rate_limit, "consume_rate_limit", _record)
    await rate_limit.throttle_auth_attempts(
        _request(peer="10.0.0.1", path="/device/v1/pairings", body=b'{"device_name":"x"}')
    )

    assert calls == [
        (["unattributable:/device/v1/pairings"], rate_limit.UNATTRIBUTABLE_AUTH_RATE_LIMIT)
    ]


async def test_attributable_attempt_never_touches_the_shared_bucket(monkeypatch) -> None:
    """The generous ceiling must not become the limit a real login is charged to."""
    from backend.users.api import rate_limit

    monkeypatch.setenv("TRUST_PEER_IP", "false")
    monkeypatch.setenv("BEHIND_PROXY", "false")
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    _clear_platform_settings()
    digest = hashlib.sha256(b"victim@example.com").hexdigest()
    calls: list[tuple[list[str], int]] = []

    async def _record(keys, *, limit, window_seconds, detail):
        calls.append((list(keys), limit))

    monkeypatch.setattr(rate_limit, "consume_rate_limit", _record)
    await rate_limit.throttle_auth_attempts(
        _request(
            peer="10.0.0.1",
            body=b'{"email":"victim@example.com","password":"x"}',
            content_type="Application/JSON",
        )
    )

    assert calls == [
        ([f"account:{digest}:/auth/login"], get_auth_settings().auth_rate_limit_requests)
    ]


async def test_client_failure_beacon_uses_the_shared_store(monkeypatch) -> None:
    """The in-process dict reset on every cold start; it enforced nothing."""
    from backend.core.api import client_failures

    monkeypatch.setenv("TRUST_PEER_IP", "false")
    _clear_platform_settings()
    calls: list[tuple[list[str], int]] = []

    async def _record(keys, *, limit, window_seconds, detail):
        calls.append((list(keys), limit))

    monkeypatch.setattr(client_failures, "consume_rate_limit", _record)
    await client_failures._throttle_client_failure(_request(peer="10.0.0.1"))

    assert calls == [(["client-failure:global"], client_failures.CLIENT_FAILURE_GLOBAL_RATE_LIMIT)]
    assert not hasattr(client_failures, "_rate_buckets")


async def test_client_failure_beacon_is_per_ip_when_the_address_is_usable(monkeypatch) -> None:
    from backend.core.api import client_failures

    monkeypatch.setenv("TRUST_PEER_IP", "true")
    _clear_platform_settings()
    calls: list[tuple[list[str], int]] = []

    async def _record(keys, *, limit, window_seconds, detail):
        calls.append((list(keys), limit))

    monkeypatch.setattr(client_failures, "consume_rate_limit", _record)
    await client_failures._throttle_client_failure(_request(peer="203.0.113.7"))

    assert calls == [(["client-failure:ip:203.0.113.7"], client_failures.CLIENT_FAILURE_RATE_LIMIT)]


def test_trust_peer_ip_defaults_to_true_for_direct_deployments() -> None:
    assert AppSettings(_env_file=None).trust_peer_ip is True


# --- Publish authorization ---
# Tests that only the project owner can expose a document publicly.


class _StubProjectRepository:
    def __init__(self, project) -> None:
        self._project = project

    async def get_by_id(self, session, project_id):
        return self._project


class _StubDocumentRepository:
    def __init__(self, document) -> None:
        self._document = document
        self.updates: list[dict] = []

    async def get_by_id(self, session, document_id):
        return self._document

    async def update(self, session, document, **fields):
        self.updates.append(fields)
        for key, value in fields.items():
            setattr(document, key, value)
        return document


def _publish_fixture(*, owner_id, collaborator_ids=()):
    from backend.document.application.document_catalog import DocumentCatalog
    from backend.document.infrastructure.orm_models import DocumentWorkflow

    class _Project:
        def __init__(self):
            self.id = uuid.uuid4()
            self.owner_id = owner_id
            self.shared_users = [type("U", (), {"id": cid})() for cid in collaborator_ids]

    class _Document:
        def __init__(self, project_id):
            self.id = uuid.uuid4()
            self.project_id = project_id
            self.workflow = DocumentWorkflow.draft
            self.name = "Codex"

    project = _Project()
    document = _Document(project.id)
    documents = _StubDocumentRepository(document)
    service = DocumentCatalog(documents=documents, projects=_StubProjectRepository(project))
    return service, project, document


async def test_collaborator_cannot_publish_a_document() -> None:
    from backend.document.infrastructure.orm_models import DocumentWorkflow

    owner_id, collaborator_id = uuid.uuid4(), uuid.uuid4()
    service, project, document = _publish_fixture(
        owner_id=owner_id, collaborator_ids=[collaborator_id]
    )
    collaborator = type("U", (), {"id": collaborator_id})()

    with pytest.raises(AccessDeniedError, match="owner"):
        await service.update_document(
            None, collaborator, project.id, document.id, workflow=DocumentWorkflow.published
        )

    assert document.workflow is DocumentWorkflow.draft
    assert service._documents.updates == []


async def test_owner_can_still_publish_a_document() -> None:
    from backend.document.infrastructure.orm_models import DocumentWorkflow

    owner_id = uuid.uuid4()
    service, project, document = _publish_fixture(
        owner_id=owner_id, collaborator_ids=[uuid.uuid4()]
    )
    owner = type("U", (), {"id": owner_id})()

    updated = await service.update_document(
        None, owner, project.id, document.id, workflow=DocumentWorkflow.published
    )

    assert updated.workflow is DocumentWorkflow.published


async def test_collaborator_can_still_rename_and_unpublish() -> None:
    """The restriction is on exposure, not on collaboration."""
    from backend.document.infrastructure.orm_models import DocumentWorkflow

    owner_id, collaborator_id = uuid.uuid4(), uuid.uuid4()
    service, project, document = _publish_fixture(
        owner_id=owner_id, collaborator_ids=[collaborator_id]
    )
    document.workflow = DocumentWorkflow.published
    collaborator = type("U", (), {"id": collaborator_id})()

    await service.update_document(None, collaborator, project.id, document.id, name="Renamed")
    await service.update_document(
        None, collaborator, project.id, document.id, workflow=DocumentWorkflow.draft
    )

    assert document.name == "Renamed"
    assert document.workflow is DocumentWorkflow.draft


async def test_orphaned_project_cannot_be_published_by_a_collaborator() -> None:
    """`is_owner` is False when owner_id is NULL, so publication stays closed."""
    from backend.document.infrastructure.orm_models import DocumentWorkflow

    collaborator_id = uuid.uuid4()
    service, project, document = _publish_fixture(owner_id=None, collaborator_ids=[collaborator_id])
    collaborator = type("U", (), {"id": collaborator_id})()

    with pytest.raises(AccessDeniedError):
        await service.update_document(
            None, collaborator, project.id, document.id, workflow=DocumentWorkflow.published
        )
