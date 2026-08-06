"""`nomicous pair` and `nomicous version`, run as the installed console script.

Everything here is live. The CLI is the real `nomicous` executable from a real
wheel in its own virtual environment; the platform is a real uvicorn process
serving the real `create_app()`; the database is real Postgres, migrated by
alembic. Nothing is patched, faked, or substituted - the approval that unblocks
each pairing is an HTTP request from a second process, exactly as a browser
would make it.

That matters more here than it usually would. Every claim this issue makes is
about behaviour *between* processes: that the URL is printed before a browser is
launched, that the code the terminal shows is the code the consent screen shows,
that the credential lands on disk readable by nobody else. A test that imported
`inference.cli.pair` and called it could not observe any of them.

Uses its own database (`kalamos_056_cli`) rather than the one
`tests/nomicous/integration/conftest.py` truncates between tests, so a server
held open across this module cannot have the ground moved under it.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT

pytestmark = pytest.mark.integration

DATABASE = "kalamos_056_cli"
POSTGRES_DSN = "postgresql://postgres:dev@localhost:5433"
APP_ORIGIN = "https://app.nomicous.test"

SERVER_START_TIMEOUT_SECONDS = 60.0
PAIRING_TIMEOUT_SECONDS = 60.0

# The real cadence would make this module spend five seconds per poll waiting for
# an approval that has already happened. The cadence itself is the platform's
# and is covered by its own tests.
POLL_INTERVAL_SECONDS = "1"


# ---------------------------------------------------------------------------
# The platform: real app, real Postgres, real HTTP
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _psql(sql: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "nomicous-db-1", "psql", "-U", "postgres", "-c", sql],
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def migrated_database() -> str:
    """Create this module's database and bring it to the current alembic head."""
    if shutil.which("docker") is None:
        pytest.skip("docker is required to reach the test Postgres")
    created = _psql(f"CREATE DATABASE {DATABASE}")
    if created.returncode != 0 and "already exists" not in created.stderr:
        pytest.skip(f"cannot reach Postgres at {POSTGRES_DSN}: {created.stderr.strip()}")

    url = f"{POSTGRES_DSN}/{DATABASE}"
    environment = dict(os.environ)
    environment.update(
        {
            "MIGRATOR_DATABASE_URL": url,
            "SYNC_DATABASE_URL": url,
            "DATABASE_URL": url.replace("postgresql://", "postgresql+asyncpg://"),
            "JWT_SECRET": "test-secret-not-for-production-at-least-32-bytes",
        }
    )
    migrated = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "infrastructure/alembic.ini", "upgrade", "head"],
        cwd=REPO_ROOT / "nomicous",
        env=environment,
        capture_output=True,
        text=True,
    )
    assert migrated.returncode == 0, migrated.stderr
    return url


@pytest.fixture(scope="session")
def platform_url(migrated_database: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """A uvicorn process serving `create_app()`, torn down at the end.

    Its output goes to a file, never to a pipe. `infrastructure/db.py` turns
    SQLAlchemy `echo` on outside production, so this server emits hundreds of
    kilobytes of SQL in the course of a few pairings - far more than a 64 KB
    pipe buffer holds. With nothing draining the pipe the server blocks on its
    own `write`, mid-request, and every later call times out against a process
    that looks alive and answers nothing.
    """
    port = _free_port()
    log_path = tmp_path_factory.mktemp("platform") / "server.log"
    environment = dict(os.environ)
    environment.update(
        {
            "MIGRATOR_DATABASE_URL": migrated_database,
            "SYNC_DATABASE_URL": migrated_database,
            "DATABASE_URL": migrated_database.replace("postgresql://", "postgresql+asyncpg://"),
            "JWT_SECRET": "test-secret-not-for-production-at-least-32-bytes",
            # A dedicated key, as production requires: keying device tokens off
            # JWT_SECRET is what ADR 0001 decision 5 refuses to allow.
            "DEVICE_TOKEN_HMAC_SECRET": "test-device-token-hmac-secret-not-for-production",
            "DEVICE_PAIRING_ENABLED": "true",
            "DEVICE_PAIRING_APP_ORIGIN": APP_ORIGIN,
            "DEVICE_PAIRING_POLL_INTERVAL_SECONDS": POLL_INTERVAL_SECONDS,
            "AUTH_RATE_LIMIT_REQUESTS": "1000",
            # Every test in this module pairs from 127.0.0.1, so they all charge
            # one per-client bucket. At the default of 10 the module exhausts it
            # partway through and the remaining tests fail on a 429 that has
            # nothing to do with what they assert.
            "DEVICE_PAIRING_RATE_LIMIT_REQUESTS": "1000",
            "JOB_WORKER_ENABLED": "false",
            "ENVIRONMENT": "development",
            # The platform imports `inference.contracts`, so the repository root
            # is on the path alongside the application package.
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "nomicous"), str(REPO_ROOT)]),
            "INFERENCE_REGISTRY_PATH": str(REPO_ROOT / "inference" / "registry.yaml"),
        }
    )
    environment.pop("INFERENCE_WORKER_SERVICE_TOKEN", None)

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
                "warning",
            ],
            cwd=REPO_ROOT / "nomicous",
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            _wait_for_health(server, base_url, log_path)
            yield base_url
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - uvicorn exits cleanly
                server.kill()


def _wait_for_health(server: subprocess.Popen, base_url: str, log_path: Path) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise AssertionError(f"the platform exited before serving:\n{log_path.read_text()}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.2)
    server.terminate()
    raise AssertionError(f"the platform did not answer {base_url}/health in time")


# ---------------------------------------------------------------------------
# The CLI: a real wheel, a real console script
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the wheel and install the console script into an empty environment.

    `--no-deps` on purpose. The published closure carries Torch, and downloading
    it would turn a CLI test into a multi-gigabyte one; the pairing path imports
    nothing from it. `rich` is installed explicitly because the CLI does import
    that, and an entry point that cannot start proves nothing. That the closure
    resolves at all is `test_published_package.py`'s question, not this one's.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build and install the CLI")

    workspace = tmp_path_factory.mktemp("cli")
    dist = workspace / "dist"
    venv = workspace / "venv"

    build = subprocess.run(
        [uv, "build", "--wheel", "-o", str(dist)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
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
    for arguments in (["--no-deps", str(wheels[0])], ["rich"]):
        installed = subprocess.run(
            [uv, "pip", "install", "--python", str(scripts / "python"), *arguments],
            capture_output=True,
            text=True,
        )
        # A wheel that builds and will not install is a real defect, not an
        # environmental one: skipping here would turn it green. `uv` being
        # absent from the machine is the only excuse, and it is handled above.
        assert installed.returncode == 0, installed.stderr

    executable = scripts / ("nomicous.exe" if os.name == "nt" else "nomicous")
    assert executable.is_file(), "the wheel did not install a `nomicous` console script"
    return {"executable": executable, "wheel": wheels[0], "workspace": workspace}


def _wheel_version(wheel: Path) -> str:
    # `nomicous_inference-0.2.0-py3-none-any.whl`
    return wheel.name.split("-")[1]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """Where this test's CLI keeps its credential. Never the real one."""
    return tmp_path / "nomicous-home"


def _cli_environment(home: Path, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    # A developer's own SSH session must not decide what these tests observe.
    for inherited in ("SSH_CONNECTION", "SSH_TTY", "BROWSER", "NOMICOUS_API_URL"):
        environment.pop(inherited, None)
    environment["NOMICOUS_HOME"] = str(home)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(extra or {})
    return environment


# ---------------------------------------------------------------------------
# Acting as the researcher's browser
# ---------------------------------------------------------------------------
def _post(url: str, body: dict, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, object]:
    request = urllib.request.Request(url, method="GET")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _delete(url: str, headers: dict[str, str]) -> int:
    request = urllib.request.Request(url, method="DELETE")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.fixture
def researcher(platform_url: str) -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    status, body = _post(
        f"{platform_url}/auth/register",
        {
            "email": f"cli-{suffix}@test.kalamos",
            "username": f"cli_{suffix}",
            "password": "test-pass-123",
        },
    )
    assert status == 201, body
    return {
        "email": f"cli-{suffix}@test.kalamos",
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


# ---------------------------------------------------------------------------
# Driving `nomicous pair` while approving it from outside
# ---------------------------------------------------------------------------
class PairRun:
    """One `nomicous pair` invocation and everything observable about it."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        verification_url: str | None,
        consent_screen: dict | None,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.verification_url = verification_url
        self.consent_screen = consent_screen
        """What `/pair` was served for this request, captured before approving.

        It has to be read then: approval consumes the pairing row, and looking
        it up afterwards is a 404 - single-use is the point of ADR 0001's
        decision 7.
        """

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"

    @property
    def confirmation_code(self) -> str | None:
        """The code as the terminal showed it, read back off the terminal."""
        lines = [line.strip() for line in self.stdout.splitlines()]
        for index, line in enumerate(lines):
            if line.startswith("Confirmation code"):
                for candidate in lines[index + 1 :]:
                    if candidate:
                        return candidate
        return None

    @property
    def verification_token(self) -> str:
        assert self.verification_url, "no pairing URL was printed"
        return self.verification_url.split("#", 1)[1]


def _run_pair(
    installed_cli: dict[str, object],
    platform_url: str,
    home: Path,
    *,
    arguments: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
    approve_as: dict[str, str] | None = None,
    deny_as: dict[str, str] | None = None,
) -> PairRun:
    """Run the real console script, and answer its consent request out of band.

    Output goes to files rather than pipes so this can read what the CLI has
    printed *while it is still running* without any chance of deadlocking on a
    full pipe buffer - the pairing URL only becomes readable partway through.
    """
    stdout_path = home.parent / f"pair-stdout-{uuid.uuid4().hex[:6]}.txt"
    stderr_path = home.parent / f"pair-stderr-{uuid.uuid4().hex[:6]}.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out_file, stderr_path.open("w") as err_file:
        process = subprocess.Popen(
            [
                str(installed_cli["executable"]),
                "pair",
                "--api-url",
                platform_url,
                *arguments,
            ],
            env=_cli_environment(home, extra=environment),
            cwd=str(installed_cli["workspace"]),
            stdout=out_file,
            stderr=err_file,
            text=True,
        )
        verification_url = None
        consent_screen = None
        try:
            if approve_as is not None or deny_as is not None:
                verification_url = _await_pairing_url(process, stdout_path)
                if verification_url is not None:
                    consent_screen = _decide(
                        platform_url,
                        verification_url,
                        headers=(approve_as or deny_as)["headers"],
                        approve=approve_as is not None,
                    )
            process.wait(timeout=PAIRING_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung CLI is the failure
            process.kill()
            process.wait()
            raise AssertionError(
                f"`nomicous pair` did not finish.\n{stdout_path.read_text()}\n"
                f"{stderr_path.read_text()}"
            ) from None

    return PairRun(
        returncode=process.returncode,
        stdout=stdout_path.read_text(),
        stderr=stderr_path.read_text(),
        verification_url=verification_url,
        consent_screen=consent_screen,
    )


def _await_pairing_url(process: subprocess.Popen, stdout_path: Path) -> str | None:
    """Read the URL off the CLI's own output, the way a researcher would."""
    deadline = time.monotonic() + PAIRING_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for line in stdout_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(APP_ORIGIN):
                return stripped
        if process.poll() is not None:
            return None
        time.sleep(0.1)
    return None


def _decide(platform_url: str, verification_url: str, *, headers: dict, approve: bool) -> dict:
    """The browser half: look the request up by its fragment token, then decide.

    Returns what the lookup served, which is exactly what the `/pair` page
    renders - including the **confirmation code** a researcher is asked to
    compare against their terminal.
    """
    token = verification_url.split("#", 1)[1]
    status, looked_up = _post(
        f"{platform_url}/devices/pairings/lookup", {"verification_token": token}, headers
    )
    assert status == 200, looked_up
    action = "approve" if approve else "deny"
    status, decided = _post(
        f"{platform_url}/devices/pairings/{looked_up['pairing_id']}/{action}",
        {"verification_token": token},
        headers,
    )
    assert status in (200, 204), decided
    return looked_up


# ---------------------------------------------------------------------------
# The pairing URL, and when the browser is allowed to be involved
# ---------------------------------------------------------------------------
def test_the_pairing_url_is_printed_before_any_browser_is_opened(
    installed_cli, platform_url, home, researcher, tmp_path
) -> None:
    """ADR 0002: the browser is a convenience layered on a printed URL.

    Proved by making `$BROWSER` a script that writes a marker onto the CLI's own
    stdout. Both writes land in one file in the order they happened, so the
    assertion is about real sequencing between two processes rather than about
    which function this module believes is called first.
    """
    stub = tmp_path / "browser-stub.sh"
    stub.write_text('#!/bin/sh\necho "BROWSER-OPENED $1"\n')
    stub.chmod(0o755)

    run = _run_pair(
        installed_cli,
        platform_url,
        home,
        environment={"BROWSER": str(stub)},
        approve_as=researcher,
    )
    assert run.returncode == 0, run.output

    lines = run.stdout.splitlines()
    url_line = next(index for index, line in enumerate(lines) if APP_ORIGIN in line)
    browser_line = next(index for index, line in enumerate(lines) if "BROWSER-OPENED" in line)

    assert url_line < browser_line, run.stdout
    # And what it was handed is the same URL the researcher was shown.
    assert lines[url_line].strip() in lines[browser_line]


def test_pairing_completes_over_ssh_with_no_browser_available(
    installed_cli, platform_url, home, researcher, tmp_path
) -> None:
    """`webbrowser.open()` over SSH opens nothing, or opens it on the wrong machine.

    The stub would record either. It records nothing, and the pairing still
    completes on the printed URL alone.
    """
    stub = tmp_path / "browser-stub.sh"
    record = tmp_path / "opened.txt"
    stub.write_text(f'#!/bin/sh\necho "$1" >> {record}\n')
    stub.chmod(0o755)

    run = _run_pair(
        installed_cli,
        platform_url,
        home,
        environment={
            "BROWSER": str(stub),
            "SSH_CONNECTION": "10.0.0.2 51000 10.0.0.9 22",
            "SSH_TTY": "/dev/pts/3",
        },
        approve_as=researcher,
    )

    assert run.returncode == 0, run.output
    assert not record.exists(), f"a browser was launched over SSH: {record.read_text()}"
    assert "SSH session" in run.stdout
    assert APP_ORIGIN in run.stdout
    assert "Paired." in run.stdout


def test_the_confirmation_code_in_the_terminal_matches_the_consent_screen(
    installed_cli, platform_url, home, researcher
) -> None:
    """ADR 0001 decision 13, finally made real on the client side.

    The consent screen's every other field is supplied by whoever started the
    pairing. This code is not, and comparing the two is the only check a
    researcher can make - so the two strings have to be the same string.
    """
    run = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert run.returncode == 0, run.output

    printed = run.confirmation_code
    assert printed, run.stdout
    assert run.consent_screen is not None

    assert printed == run.consent_screen["confirmation_code"]
    # It is shown before the wait begins, not after approval - a code that
    # arrives once the researcher has already clicked checks nothing.
    assert run.stdout.index(printed) < run.stdout.index("Paired.")


# ---------------------------------------------------------------------------
# The credential on disk
# ---------------------------------------------------------------------------
def test_the_credential_is_written_with_owner_only_permissions(
    installed_cli, platform_url, home, researcher
) -> None:
    """ADR 0001: `0600` in a `0700` directory, so another account cannot claim as them."""
    run = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert run.returncode == 0, run.output

    credential = home / "device.json"
    assert credential.is_file(), run.output
    assert stat.S_IMODE(credential.stat().st_mode) == 0o600
    assert stat.S_IMODE(home.stat().st_mode) == 0o700

    stored = json.loads(credential.read_text())
    assert stored["platform_url"] == platform_url
    assert stored["account_email"] == researcher["email"]
    assert stored["device_token"].startswith("nmd1.")
    # The credential is the only place the token exists. It is never printed.
    assert stored["device_token"] not in run.output


def test_the_stored_token_actually_authenticates_this_machine(
    installed_cli, platform_url, home, researcher
) -> None:
    """The point of the file: what `nomicous run` will present in #57."""
    run = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert run.returncode == 0, run.output
    stored = json.loads((home / "device.json").read_text())

    status, identity = _get(
        f"{platform_url}/device/v1/self",
        {"X-Nomicous-Device-Token": stored["device_token"]},
    )

    assert status == 200, identity
    assert identity["device_id"] == stored["device_id"]
    assert identity["account_email"] == researcher["email"]


# ---------------------------------------------------------------------------
# The two states that end without a new device row
# ---------------------------------------------------------------------------
def test_pairing_an_already_paired_machine_reports_it_and_creates_no_second_device(
    installed_cli, platform_url, home, researcher
) -> None:
    first = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert first.returncode == 0, first.output

    # No approval offered: if this started a pairing it would sit there polling.
    second = _run_pair(installed_cli, platform_url, home)

    assert second.returncode == 0, second.output
    assert "already paired" in second.stdout.lower()
    assert APP_ORIGIN not in second.output, "a second pairing request was started"

    status, devices = _get(f"{platform_url}/devices", researcher["headers"])
    assert status == 200
    assert len(devices) == 1, devices


def test_a_revoked_device_reports_the_revocation_and_exits_non_zero(
    installed_cli, platform_url, home, researcher
) -> None:
    """Revocation is a decision made on the account (ADR 0001, decision 11).

    Re-pairing over it silently would undo it without telling anyone, and
    polling for a consent nobody is going to give would hang. It does neither.
    """
    paired = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert paired.returncode == 0, paired.output
    device_id = json.loads((home / "device.json").read_text())["device_id"]

    assert _delete(f"{platform_url}/devices/{device_id}", researcher["headers"]) == 204

    started = time.monotonic()
    revoked = _run_pair(installed_cli, platform_url, home)
    elapsed = time.monotonic() - started

    assert revoked.returncode != 0, revoked.output
    assert "removed" in revoked.stderr.lower()
    assert "--force" in revoked.stderr
    # It reported and stopped. Nothing was started that could be waited on.
    assert APP_ORIGIN not in revoked.output
    assert elapsed < 20, f"it spun for {elapsed:.1f}s instead of reporting"


def test_force_pairs_again_over_a_revoked_credential(
    installed_cli, platform_url, home, researcher
) -> None:
    """The recovery the refusal points at has to work."""
    paired = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert paired.returncode == 0, paired.output
    first_device = json.loads((home / "device.json").read_text())["device_id"]
    assert _delete(f"{platform_url}/devices/{first_device}", researcher["headers"]) == 204

    repaired = _run_pair(
        installed_cli, platform_url, home, arguments=("--force",), approve_as=researcher
    )

    assert repaired.returncode == 0, repaired.output
    assert json.loads((home / "device.json").read_text())["device_id"] != first_device


def test_a_denied_pairing_exits_non_zero_and_writes_nothing(
    installed_cli, platform_url, home, researcher
) -> None:
    run = _run_pair(installed_cli, platform_url, home, deny_as=researcher)

    assert run.returncode != 0, run.output
    assert "refused" in run.stderr.lower()
    assert not (home / "device.json").exists()


def test_a_platform_that_cannot_be_reached_is_not_reported_as_a_revocation(
    installed_cli, platform_url, home, researcher
) -> None:
    """A researcher on a train has not been revoked, and must not be told so."""
    paired = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert paired.returncode == 0, paired.output

    unreachable = f"http://127.0.0.1:{_free_port()}"
    run = _run_pair(installed_cli, unreachable, home)

    assert run.returncode != 0
    assert "removed" not in run.stderr.lower()
    assert "revoked" not in run.stderr.lower()
    # Still paired, and the credential was left exactly as it was.
    assert json.loads((home / "device.json").read_text())["platform_url"] == platform_url


# ---------------------------------------------------------------------------
# `nomicous version`
# ---------------------------------------------------------------------------
def test_the_version_subcommand_reports_the_installed_package_version(installed_cli, home) -> None:
    """Read from installed metadata, so it is the version that is really on disk."""
    completed = subprocess.run(
        [str(installed_cli["executable"]), "version"],
        env=_cli_environment(home),
        cwd=str(installed_cli["workspace"]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = _wheel_version(installed_cli["wheel"])
    assert f"nomicous-inference {expected}" in completed.stdout
    # Not a source checkout: the whole point of reading metadata.
    assert "0+unknown" not in completed.stdout


def test_the_version_subcommand_names_the_header_the_platform_reads(installed_cli, home) -> None:
    """The **version floor** refuses on this exact string (issue 055)."""
    completed = subprocess.run(
        [str(installed_cli["executable"]), "version"],
        env=_cli_environment(home),
        cwd=str(installed_cli["workspace"]),
        capture_output=True,
        text=True,
    )

    expected = _wheel_version(installed_cli["wheel"])
    assert f"X-Nomicous-Agent-Version: {expected}" in completed.stdout


def test_the_version_subcommand_reports_whether_this_machine_is_paired(
    installed_cli, platform_url, home, researcher
) -> None:
    unpaired = subprocess.run(
        [str(installed_cli["executable"]), "version"],
        env=_cli_environment(home),
        cwd=str(installed_cli["workspace"]),
        capture_output=True,
        text=True,
    )
    assert "nomicous pair" in unpaired.stdout

    paired_run = _run_pair(installed_cli, platform_url, home, approve_as=researcher)
    assert paired_run.returncode == 0, paired_run.output

    paired = subprocess.run(
        [str(installed_cli["executable"]), "version"],
        env=_cli_environment(home),
        cwd=str(installed_cli["workspace"]),
        capture_output=True,
        text=True,
    )

    assert researcher["email"] in paired.stdout
    assert platform_url in paired.stdout
    # A version report must never be a way to read the credential out.
    stored = json.loads((home / "device.json").read_text())
    assert stored["device_token"] not in paired.stdout
