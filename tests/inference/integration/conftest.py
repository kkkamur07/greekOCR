"""Scaffolding the three CLI integration modules had each written for themselves.

Nothing here is a fixture in the pytest sense except by a module opting in: these
are the plumbing primitives - a free port, a migrated database, a uvicorn
process, a built wheel, a hand-rolled HTTP client - that
``test_cli_pairing.py``, ``test_cli_run.py`` and ``test_cli_self_upgrade.py``
all need and all used to carry a near-identical copy of.

The split is deliberate. What is *shared* is the mechanism; what stays in each
module is the policy, because the three platforms differ in ways that matter and
must keep differing: one sets a **version floor**, one serves page images out of
a local media root with a **service credential** configured, one raises the
pairing rate limit because every test in it pairs from ``127.0.0.1``. So
``platform_environment`` supplies only the settings all three hold identically
and takes the rest as overrides, rather than growing a union of every knob any
one module ever needed.

Each module keeps its own database, for the reason each of them documented
separately: a server held open across a module cannot have the ground moved
under it by ``tests/nomicous/integration/conftest.py``'s truncation.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT

POSTGRES_DSN = "postgresql://postgres:dev@localhost:5433"
APP_ORIGIN = "https://app.nomicous.test"
JWT_SECRET = "test-secret-not-for-production-at-least-32-bytes"
#: A dedicated key, as production requires: keying device tokens off
#: ``JWT_SECRET`` is what ADR 0001 decision 5 refuses to allow.
DEVICE_TOKEN_HMAC_SECRET = "test-device-token-hmac-secret-not-for-production"

SERVER_START_TIMEOUT_SECONDS = 60.0
PAIRING_TIMEOUT_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Tools this suite cannot substitute for
# ---------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def require_uv() -> str:
    """`uv` being absent is the one environmental excuse these modules accept."""
    executable = shutil.which("uv")
    if executable is None:
        pytest.skip("uv is required to build and install the CLI")
    return executable


# ---------------------------------------------------------------------------
# Postgres, and one database per module
# ---------------------------------------------------------------------------
def psql(sql: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "nomicous-db-1", "psql", "-U", "postgres", "-c", sql],
        capture_output=True,
        text=True,
    )


def migrate_database(database: str) -> str:
    """Create *database* and bring it to the current alembic head."""
    if shutil.which("docker") is None:
        pytest.skip("docker is required to reach the test Postgres")
    created = psql(f"CREATE DATABASE {database}")
    if created.returncode != 0 and "already exists" not in created.stderr:
        pytest.skip(f"cannot reach Postgres at {POSTGRES_DSN}: {created.stderr.strip()}")

    url = f"{POSTGRES_DSN}/{database}"
    environment = dict(os.environ)
    environment.update(database_environment(url))
    environment["JWT_SECRET"] = JWT_SECRET
    migrated = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "infrastructure/alembic.ini", "upgrade", "head"],
        cwd=REPO_ROOT / "nomicous",
        env=environment,
        capture_output=True,
        text=True,
    )
    assert migrated.returncode == 0, migrated.stderr
    return url


def database_environment(url: str) -> dict[str, str]:
    return {
        "MIGRATOR_DATABASE_URL": url,
        "SYNC_DATABASE_URL": url,
        "DATABASE_URL": url.replace("postgresql://", "postgresql+asyncpg://"),
    }


# ---------------------------------------------------------------------------
# The platform: a real uvicorn process on the real `create_app()`
# ---------------------------------------------------------------------------
def wait_for_http(
    process: subprocess.Popen,
    url: str,
    *,
    what: str,
    log_path: Path | None = None,
) -> None:
    """Poll *url* until it answers 200, or say why it never will."""
    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = f":\n{log_path.read_text()}" if log_path is not None else ""
            raise AssertionError(f"{what} exited before serving{detail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.2)
    process.terminate()
    raise AssertionError(f"{what} did not answer {url} in time")


def platform_environment(database_url: str, **overrides: str) -> dict[str, str]:
    """The settings all three modules' platforms hold identically, plus overrides.

    ``INFERENCE_WORKER_SERVICE_TOKEN`` is *popped* rather than defaulted: a
    developer's ambient value must not decide whether a module's platform accepts
    a **service credential**. The one module that wants one passes it back in.
    """
    environment = dict(os.environ)
    environment.pop("INFERENCE_WORKER_SERVICE_TOKEN", None)
    environment.update(database_environment(database_url))
    environment.update(
        {
            "JWT_SECRET": JWT_SECRET,
            "DEVICE_TOKEN_HMAC_SECRET": DEVICE_TOKEN_HMAC_SECRET,
            "DEVICE_PAIRING_ENABLED": "true",
            "DEVICE_PAIRING_APP_ORIGIN": APP_ORIGIN,
            "AUTH_RATE_LIMIT_REQUESTS": "1000",
            # The platform's own worker must not race an agent for pages. With
            # ADR 0003 it has no way to run them anyway; this makes that explicit
            # rather than relying on it.
            "JOB_WORKER_ENABLED": "false",
            "ENVIRONMENT": "development",
            # The platform imports `inference.contracts`, so the repository root
            # is on the path alongside the application package.
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "nomicous"), str(REPO_ROOT)]),
            "INFERENCE_REGISTRY_PATH": str(REPO_ROOT / "inference" / "registry.yaml"),
        }
    )
    environment.update(overrides)
    return environment


def start_platform(
    database_url: str,
    log_path: Path,
    *,
    what: str = "the platform",
    log_level: str = "warning",
    **overrides: str,
) -> tuple[subprocess.Popen, str]:
    """Start one uvicorn process and return it with its base URL, once healthy.

    Output goes to a file, never a pipe. `infrastructure/db.py` turns SQLAlchemy
    `echo` on outside production, so a server that runs a few pages emits far
    more than a 64 KB pipe buffer holds; with nothing draining it the server
    blocks on its own `write` mid-request and every later call times out against
    a process that looks alive and answers nothing.

    The parent's handle on that file is closed as soon as the child is spawned -
    `Popen` gives the child its own descriptor, so nothing is lost and no handle
    is left open for the life of the module.
    """
    port = free_port()
    with log_path.open("w") as log_file:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.core.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                log_level,
            ],
            cwd=REPO_ROOT / "nomicous",
            env=platform_environment(database_url, **overrides),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    base_url = f"http://127.0.0.1:{port}"
    wait_for_http(server, f"{base_url}/health", what=what, log_path=log_path)
    return server, base_url


def stop_platform(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover - uvicorn exits cleanly
        server.kill()


def serve_platform(
    database_url: str,
    log_path: Path,
    *,
    log_level: str = "warning",
    **overrides: str,
) -> Iterator[str]:
    """`start_platform` as a fixture body: yield the base URL, then tear down."""
    server, base_url = start_platform(database_url, log_path, log_level=log_level, **overrides)
    try:
        yield base_url
    finally:
        stop_platform(server)


# ---------------------------------------------------------------------------
# The CLI: a real wheel, a real console script, an empty environment
# ---------------------------------------------------------------------------
def build_and_install_cli(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    install_sets: Sequence[Sequence[str]],
) -> dict[str, object]:
    """Build the wheel and install it into a fresh venv, one `install_sets` entry
    at a time.

    The wheel path is substituted for the literal ``"{wheel}"`` in any argument,
    so a caller can choose `--no-deps` or the full closure without this having an
    opinion about which is right for it.

    A failed install is a failure, never a skip: a wheel that builds and will not
    install is exactly the defect this is positioned to catch. `uv` being absent
    from the machine is the only environmental excuse, and `require_uv` handles it.
    """
    uv = require_uv()
    workspace = tmp_path_factory.mktemp("cli")
    dist = workspace / "dist"
    venv = workspace / "venv"

    build = subprocess.run(
        [uv, "build", "--wheel", "-o", str(dist)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert build.returncode == 0, build.stderr
    wheels = sorted(dist.glob("nomicous_inference-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    subprocess.run(
        [uv, "venv", str(venv), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
        check=True,
        capture_output=True,
        text=True,
    )
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    for arguments in install_sets:
        resolved = [str(wheels[0]) if argument == "{wheel}" else argument for argument in arguments]
        installed = subprocess.run(
            [uv, "pip", "install", "--python", str(scripts / "python"), *resolved],
            capture_output=True,
            text=True,
        )
        assert installed.returncode == 0, installed.stderr

    executable = scripts / ("nomicous.exe" if os.name == "nt" else "nomicous")
    assert executable.is_file(), "the wheel did not install a `nomicous` console script"
    return {"executable": executable, "wheel": wheels[0], "workspace": workspace}


def cli_environment(
    home: Path,
    *,
    drop: Sequence[str] = (),
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment the installed CLI runs under.

    *drop* is per-module and deliberately not defaulted to a union: `run` has to
    drop `INFERENCE_REGISTRY_PATH` or the installed package reads the **Registry**
    out of the checkout instead of its own bundled copy, and a module that does
    not drop it is making a different, equally deliberate choice.
    """
    environment = dict(os.environ)
    for inherited in drop:
        environment.pop(inherited, None)
    environment["NOMICOUS_HOME"] = str(home)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(extra or {})
    return environment


# ---------------------------------------------------------------------------
# Acting as the researcher, and as the researcher's browser
# ---------------------------------------------------------------------------
def _decoded(raw: bytes) -> object:
    try:
        return json.loads(raw or b"{}")
    except json.JSONDecodeError:  # pragma: no cover - only on a non-JSON error body
        return {}


def http_request(
    method: str,
    url: str,
    body: dict | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[int, object]:
    """One request, with the status and the decoded body of *whatever* came back.

    An `HTTPError` is an answer here, not an exception: these modules assert on
    422s and 426s as often as on 200s.
    """
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _decoded(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decoded(exc.read())


def register_account(platform_url: str, prefix: str) -> tuple[str, dict[str, str]]:
    """A fresh account, and the bearer header that acts as it."""
    suffix = uuid.uuid4().hex[:8]
    email = f"{prefix}-{suffix}@test.kalamos"
    status, body = http_request(
        "POST",
        f"{platform_url}/auth/register",
        {"email": email, "username": f"{prefix}_{suffix}", "password": "test-pass-123"},
    )
    assert status == 201, body
    return email, {"Authorization": f"Bearer {body['access_token']}"}


def await_line(
    process: subprocess.Popen,
    path: Path,
    prefix: str,
    *,
    timeout: float = PAIRING_TIMEOUT_SECONDS,
) -> str | None:
    """Read a line off a running process's output file, as a researcher would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in path.read_text().splitlines():
            if line.strip().startswith(prefix):
                return line.strip()
        if process.poll() is not None:
            return None
        time.sleep(0.1)
    return None


def decide_pairing(
    platform_url: str,
    verification_url: str,
    *,
    headers: dict,
    approve: bool = True,
) -> dict:
    """The browser half: look the request up by its fragment token, then decide.

    Returns what the lookup served, which is exactly what the `/pair` page
    renders - including the **confirmation code** a researcher is asked to
    compare against their terminal. It has to be read *before* deciding:
    approval consumes the pairing row and looking it up afterwards is a 404,
    single-use being the point of ADR 0001's decision 7.
    """
    token = verification_url.split("#", 1)[1]
    status, looked_up = http_request(
        "POST", f"{platform_url}/devices/pairings/lookup", {"verification_token": token}, headers
    )
    assert status == 200, looked_up
    action = "approve" if approve else "deny"
    status, decided = http_request(
        "POST",
        f"{platform_url}/devices/pairings/{looked_up['pairing_id']}/{action}",
        {"verification_token": token},
        headers,
    )
    assert status in (200, 204), decided
    return looked_up
