"""`nomicous run`, the **claim** loop, driven as the installed console script.

Everything here is live, and on this issue that is not a stylistic preference.
Every claim the run loop makes is about behaviour *between* processes: that a
page is claimed over HTTP and reported back over HTTP, that exactly one is in
flight at a time, that a Ctrl-C delivered to a real process ends the page it was
holding, that a `SIGKILL`ed process leaves a page the platform's **lease**
later releases. None of that is observable from inside one interpreter, and a
test that imported `inference.cli.run` and called it would be asserting about
its own mocks.

So: the CLI is the real `nomicous` executable from a real wheel with its real
dependency closure installed, the platform is a real uvicorn process serving the
real `create_app()`, the database is real Postgres migrated by alembic, and the
models are the real **Hub artifact**s - PyTorch checkpoints, resolved through the
**Hub cache** exactly as a researcher's laptop resolves them (ADR 0004; the
issue's mention of ONNX predates it).

Three platforms, because three of the behaviours under test are settings the
platform owns and cannot hold two values of at once: the **version floor** that
refuses an agent, the short **lease** that releases a killed agent's page, and
the ordinary configuration everything else runs against.

Uses its own databases rather than the one `tests/nomicous/integration/conftest.py`
truncates between tests, so a server held open across this module cannot have the
ground moved under it.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT, SEGMENT_PAGE

pytestmark = pytest.mark.integration

DATABASE = "kalamos_057_run"
LEASE_DATABASE = "kalamos_057_lease"
POSTGRES_DSN = "postgresql://postgres:dev@localhost:5433"
APP_ORIGIN = "https://app.nomicous.test"

SERVER_START_TIMEOUT_SECONDS = 60.0
PAIRING_TIMEOUT_SECONDS = 60.0
#: A page load plus a real model run, from a cold process. Generous on purpose:
#: an assertion about how long Torch takes is an assertion about the machine.
RUN_TIMEOUT_SECONDS = 600.0

SERVICE_TOKEN = "test-inference-worker-service-token-not-for-production"

#: The floor `refusing_platform_url` puts above every version that exists, so a
#: current agent is refused there and nowhere else.
IMPOSSIBLE_MINIMUM_VERSION = "99.0.0"
#: Above the wheel's version but not the floor, so every claim carries an
#: **outdated** notice - a different state from being refused.
NEWER_THAN_THIS_AGENT = "9.9.9"

#: The smallest lease `DeviceSettings` allows. Short enough for a test to
#: outlive, long enough that the platform still calls it a lease.
LEASE_SECONDS = 30


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


def _migrate(database: str) -> str:
    """Create *database* and bring it to the current alembic head."""
    if shutil.which("docker") is None:
        pytest.skip("docker is required to reach the test Postgres")
    created = _psql(f"CREATE DATABASE {database}")
    if created.returncode != 0 and "already exists" not in created.stderr:
        pytest.skip(f"cannot reach Postgres at {POSTGRES_DSN}: {created.stderr.strip()}")

    url = f"{POSTGRES_DSN}/{database}"
    environment = dict(os.environ)
    environment.update(_database_environment(url))
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


JWT_SECRET = "test-secret-not-for-production-at-least-32-bytes"


def _database_environment(url: str) -> dict[str, str]:
    return {
        "MIGRATOR_DATABASE_URL": url,
        "SYNC_DATABASE_URL": url,
        "DATABASE_URL": url.replace("postgresql://", "postgresql+asyncpg://"),
    }


def _serve(database_url: str, media_root: Path, log_path: Path, **overrides: str):
    """Start one uvicorn process on `create_app()` and yield its base URL.

    Output goes to a file, never a pipe. `infrastructure/db.py` turns SQLAlchemy
    `echo` on outside production, so a server that runs a few pages emits far
    more than a 64 KB pipe buffer holds; with nothing draining it the server
    blocks on its own `write` mid-request and every later call times out against
    a process that looks alive and answers nothing.
    """
    port = _free_port()
    environment = dict(os.environ)
    environment.update(_database_environment(database_url))
    environment.update(
        {
            "JWT_SECRET": JWT_SECRET,
            # A dedicated key, as production requires: keying device tokens off
            # JWT_SECRET is what ADR 0001 decision 5 refuses to allow.
            "DEVICE_TOKEN_HMAC_SECRET": "test-device-token-hmac-secret-not-for-production",
            "DEVICE_PAIRING_ENABLED": "true",
            "DEVICE_PAIRING_APP_ORIGIN": APP_ORIGIN,
            "DEVICE_PAIRING_POLL_INTERVAL_SECONDS": "1",
            "AUTH_RATE_LIMIT_REQUESTS": "1000",
            # The platform's own worker must not race the agent for pages. With
            # ADR 0003 it has no way to run them anyway; this makes that explicit
            # rather than relying on it.
            "JOB_WORKER_ENABLED": "false",
            "ENVIRONMENT": "development",
            # Pinned, never inherited. Settings read an ambient dotenv, and a
            # developer's `.env` pointing at a live Supabase project would send
            # this suite's page images somewhere real.
            "STORAGE_BACKEND": "local",
            "MEDIA_ROOT": str(media_root),
            "INFERENCE_WORKER_SERVICE_TOKEN": SERVICE_TOKEN,
            # The platform imports `inference.contracts`, so the repository root
            # is on the path alongside the application package.
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "nomicous"), str(REPO_ROOT)]),
            "INFERENCE_REGISTRY_PATH": str(REPO_ROOT / "inference" / "registry.yaml"),
        }
    )
    environment.update(overrides)

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


@pytest.fixture(scope="session")
def media_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def migrated_database() -> str:
    return _migrate(DATABASE)


@pytest.fixture(scope="session")
def platform_url(migrated_database, media_root, tmp_path_factory):
    """The platform everything ordinary runs against.

    `INFERENCE_AGENT_LATEST_VERSION` is set above the wheel's version so every
    claim here carries an **outdated** notice. That is the served state, not the
    refused one, and having it on by default proves the loop keeps working while
    it is being told to upgrade.
    """
    log_path = tmp_path_factory.mktemp("platform") / "server.log"
    yield from _serve(
        migrated_database,
        media_root,
        log_path,
        INFERENCE_AGENT_LATEST_VERSION=NEWER_THAN_THIS_AGENT,
    )


@pytest.fixture(scope="session")
def refusing_platform_url(migrated_database, media_root, tmp_path_factory):
    """A platform whose **version floor** is above every version that exists.

    Its own process rather than a setting flipped mid-suite: the floor is read
    per request from process settings, and one server cannot hold two values.
    """
    log_path = tmp_path_factory.mktemp("refusing") / "server.log"
    yield from _serve(
        migrated_database,
        media_root,
        log_path,
        INFERENCE_AGENT_MIN_VERSION=IMPOSSIBLE_MINIMUM_VERSION,
        INFERENCE_AGENT_LATEST_VERSION=IMPOSSIBLE_MINIMUM_VERSION,
    )


@pytest.fixture(scope="session")
def short_lease_platform_url(media_root, tmp_path_factory):
    """A platform with the shortest **lease** its settings allow, on its own database.

    Its own database because the stale sweep is global to whichever process runs
    it: a 30-second lease sweeping the shared database could re-queue a page one
    of the other tests was legitimately holding.
    """
    log_path = tmp_path_factory.mktemp("short-lease") / "server.log"
    yield from _serve(
        _migrate(LEASE_DATABASE),
        media_root,
        log_path,
        DEVICE_LEASE_SECONDS=str(LEASE_SECONDS),
        # The sweep is throttled per process so a hot endpoint cannot become a
        # sweep loop. Thirty seconds of that on top of a thirty-second lease
        # would make this test a race against a cost control, so the dial is
        # turned down rather than the wait padded out.
        JOB_STALE_SWEEP_MIN_INTERVAL_SECONDS="0.1",
    )


# ---------------------------------------------------------------------------
# The CLI: a real wheel, its real closure, a real console script
# ---------------------------------------------------------------------------
#: ADR 0004 requires the CPU-only Torch build. On Linux the default PyPI wheel
#: pulls sixteen nvidia/triton packages behind `torch`; without this flag the
#: fixture alone installs about 4.8 GB.
CPU_TORCH_FLAG = "--torch-backend=cpu"


@pytest.fixture(scope="session")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the wheel and install it, closure and all, into an empty environment.

    Unlike `test_cli_pairing.py` this cannot use `--no-deps`: the whole point of
    `run` is that it executes a model, so the run loop's dependency closure - and
    Torch in particular - is part of what is under test. Nothing here is
    satisfied from the repository tree.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build and install the CLI")

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
    installed = subprocess.run(
        [uv, "pip", "install", "--python", str(scripts / "python"), CPU_TORCH_FLAG, str(wheels[0])],
        capture_output=True,
        text=True,
    )
    if installed.returncode != 0:
        pytest.skip(f"cannot install the CLI: {installed.stderr.strip()}")

    executable = scripts / ("nomicous.exe" if os.name == "nt" else "nomicous")
    assert executable.is_file(), "the wheel did not install a `nomicous` console script"
    return {"executable": executable, "wheel": wheels[0], "workspace": workspace}


def _cli_environment(home: Path, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    # The suite's own repository-relative settings must not reach the installed
    # package: `INFERENCE_REGISTRY_PATH` in particular would have it read the
    # **Registry** out of the checkout instead of its own bundled copy.
    for inherited in (
        "PYTHONPATH",
        "INFERENCE_REGISTRY_PATH",
        "SSH_CONNECTION",
        "SSH_TTY",
        "BROWSER",
        "NOMICOUS_API_URL",
        "NOMICOUS_SERVICE_TOKEN",
        "NOMICOUS_WORKER_NAME",
    ):
        environment.pop(inherited, None)
    environment["NOMICOUS_HOME"] = str(home)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(extra or {})
    return environment


# ---------------------------------------------------------------------------
# Acting as the researcher's browser and as the researcher
# ---------------------------------------------------------------------------
def _request(
    method: str, url: str, body: dict | None = None, headers: dict[str, str] | None = None
) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _post(url: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, object]:
    return _request("POST", url, body, headers)


def _get(url: str, headers: dict | None = None) -> tuple[int, object]:
    return _request("GET", url, None, headers)


def _put(url: str, body: dict, headers: dict) -> tuple[int, object]:
    return _request("PUT", url, body, headers)


def _upload_part(base: str, document_id: str, headers: dict[str, str], image: bytes) -> str:
    """Multipart upload, hand-built. The stdlib has no client for it, and adding
    an HTTP library to reach one endpoint is not worth it."""
    boundary = f"----nomicous{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="page.jpeg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            image,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{base}/{document_id}/parts", data=body, method="POST", headers=dict(headers)
    )
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["id"]


class Researcher:
    """One account, one project, one paired machine, ready to be given work."""

    def __init__(self, platform_url: str, headers: dict[str, str], email: str, home: Path):
        self.platform_url = platform_url
        self.headers = headers
        self.email = email
        self.home = home
        slug = f"run-loop-{uuid.uuid4().hex[:8]}"
        status, project = _post(
            f"{platform_url}/projects", {"name": "Run loop", "slug": slug}, headers
        )
        assert status == 201, project
        self.project_id = project["id"]

    @property
    def documents_url(self) -> str:
        return f"{self.platform_url}/projects/{self.project_id}/documents"

    def new_page(self, image: bytes = b"") -> tuple[str, str]:
        status, document = _post(self.documents_url, {"name": "Claimable page"}, self.headers)
        assert status == 201, document
        part_id = _upload_part(
            self.documents_url, document["id"], self.headers, image or SEGMENT_PAGE.read_bytes()
        )
        return document["id"], part_id

    def submit_segment(self, ids: tuple[str, str]) -> str:
        document_id, part_id = ids
        status, body = _post(
            f"{self.documents_url}/{document_id}/parts/{part_id}/segment", None, self.headers
        )
        assert status == 202, body
        return body["job_id"]

    def prefer_local(self) -> None:
        """Without this the job is routed to `cloud` and no device token may
        claim it - the **execution target** is fixed at submission."""
        status, body = _put(
            f"{self.platform_url}/account/execution-target",
            {"prefer_local_inference": True},
            self.headers,
        )
        assert status == 200, body

    def job(self, job_id: str) -> dict:
        status, body = _get(f"{self.platform_url}/jobs/{job_id}", self.headers)
        assert status == 200, body
        return body

    def await_job(self, job_id: str, *, status: str, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        seen = None
        while time.monotonic() < deadline:
            seen = self.job(job_id)
            if seen["status"] == status:
                return seen
            time.sleep(0.2)
        raise AssertionError(f"job {job_id} is {seen and seen['status']!r}, not {status!r}")

    def lines(self, ids: tuple[str, str]) -> list[dict]:
        document_id, part_id = ids
        status, body = _get(
            f"{self.documents_url}/{document_id}/parts/{part_id}/lines", self.headers
        )
        assert status == 200, body
        return body


def _register(platform_url: str, home: Path) -> Researcher:
    suffix = uuid.uuid4().hex[:8]
    email = f"run-{suffix}@test.kalamos"
    status, body = _post(
        f"{platform_url}/auth/register",
        {"email": email, "username": f"run_{suffix}", "password": "test-pass-123"},
    )
    assert status == 201, body
    return Researcher(
        platform_url, {"Authorization": f"Bearer {body['access_token']}"}, email, home
    )


# ---------------------------------------------------------------------------
# Pairing this machine, through the real `nomicous pair`
# ---------------------------------------------------------------------------
def _pair(installed_cli: dict, platform_url: str, researcher: Researcher) -> str:
    """Authorise `researcher.home` by running the console script and approving it.

    Through the CLI rather than by writing `device.json`, because what `run`
    reads has to be what `pair` writes - that seam is the product, and a
    hand-built credential file would not test it.
    """
    stdout_path = researcher.home.parent / f"pair-{uuid.uuid4().hex[:6]}.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out_file:
        process = subprocess.Popen(
            [str(installed_cli["executable"]), "pair", "--api-url", platform_url, "--no-browser"],
            env=_cli_environment(researcher.home),
            cwd=str(installed_cli["workspace"]),
            stdout=out_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            url = _await_line(process, stdout_path, APP_ORIGIN)
            assert url is not None, stdout_path.read_text()
            _approve(platform_url, url, researcher.headers)
            process.wait(timeout=PAIRING_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung CLI is the failure
            process.kill()
            raise AssertionError(f"`nomicous pair` hung:\n{stdout_path.read_text()}") from None

    assert process.returncode == 0, stdout_path.read_text()
    return json.loads((researcher.home / "device.json").read_text())["device_id"]


def _await_line(process: subprocess.Popen, path: Path, prefix: str) -> str | None:
    """Read a line off a running process's output file, as a researcher would."""
    deadline = time.monotonic() + PAIRING_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for line in path.read_text().splitlines():
            if line.strip().startswith(prefix):
                return line.strip()
        if process.poll() is not None:
            return None
        time.sleep(0.1)
    return None


def _approve(platform_url: str, verification_url: str, headers: dict) -> None:
    token = verification_url.split("#", 1)[1]
    status, looked_up = _post(
        f"{platform_url}/devices/pairings/lookup", {"verification_token": token}, headers
    )
    assert status == 200, looked_up
    status, decided = _post(
        f"{platform_url}/devices/pairings/{looked_up['pairing_id']}/approve",
        {"verification_token": token},
        headers,
    )
    assert status in (200, 204), decided


def _announce_capacity(platform_url: str, home: Path) -> None:
    """Report **capacity** the way the agent does: by asking for work.

    Submission refuses to create a `local` page when no device for that host was
    seen recently, so this has to happen before anything is queued.
    """
    token = json.loads((home / "device.json").read_text())["device_token"]
    status, body = _post(
        f"{platform_url}/device/v1/jobs/claim",
        {"wait_seconds": 0},
        {"X-Nomicous-Device-Token": token, "X-Nomicous-Agent-Version": "1.0.0"},
    )
    assert status == 200, body


@pytest.fixture
def agent(installed_cli, platform_url, tmp_path) -> Researcher:
    """A registered researcher with this machine paired and ready to take work."""
    home = tmp_path / "nomicous-home"
    researcher = _register(platform_url, home)
    _pair(installed_cli, platform_url, researcher)
    _announce_capacity(platform_url, home)
    researcher.prefer_local()
    return researcher


# ---------------------------------------------------------------------------
# Driving `nomicous run`
# ---------------------------------------------------------------------------
class RunProcess:
    """One `nomicous run` invocation, readable while it is still running."""

    def __init__(self, process: subprocess.Popen, stdout: Path, stderr: Path):
        self.process = process
        self.stdout_path = stdout
        self.stderr_path = stderr

    @property
    def stdout(self) -> str:
        return self.stdout_path.read_text()

    @property
    def stderr(self) -> str:
        return self.stderr_path.read_text()

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def await_output(self, needle: str, *, timeout: float = RUN_TIMEOUT_SECONDS) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if needle in self.stdout:
                return True
            if self.process.poll() is not None:
                return needle in self.stdout
            time.sleep(0.1)
        return False

    def wait(self, timeout: float = RUN_TIMEOUT_SECONDS) -> int:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung loop is the failure
            self.process.kill()
            self.process.wait()
            raise AssertionError(f"`nomicous run` did not finish:\n{self.output}") from None


def _start_run(
    installed_cli: dict,
    platform_url: str,
    home: Path,
    *,
    arguments: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
) -> RunProcess:
    """Start the console script with its output on files rather than pipes.

    Files because this has to read what the CLI has printed *while it is still
    running* - to know when a page has been claimed, so a signal can be sent at a
    meaningful moment - and because a loop that runs several pages prints more
    than a pipe nobody is draining will hold.
    """
    marker = uuid.uuid4().hex[:6]
    stdout_path = home.parent / f"run-stdout-{marker}.txt"
    stderr_path = home.parent / f"run-stderr-{marker}.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("")
    stderr_path.write_text("")

    with stdout_path.open("w") as out_file, stderr_path.open("w") as err_file:
        process = subprocess.Popen(
            [str(installed_cli["executable"]), "run", "--api-url", platform_url, *arguments],
            env=_cli_environment(home, extra=environment),
            cwd=str(installed_cli["workspace"]),
            stdout=out_file,
            stderr=err_file,
            text=True,
        )
    return RunProcess(process, stdout_path, stderr_path)


def _run(
    installed_cli: dict,
    platform_url: str,
    home: Path,
    *,
    arguments: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
) -> RunProcess:
    started = _start_run(
        installed_cli, platform_url, home, arguments=arguments, environment=environment
    )
    started.wait()
    return started


def _broken_registry(path: Path) -> Path:
    """A **Registry** whose segmenter points at a weight file this machine does
    not have - the ordinary way a page fails to run on one particular laptop."""
    path.write_text(
        "models:\n"
        "  blla-segment:\n"
        "    task: segment\n"
        "    architecture: blla\n"
        "    device: cpu\n"
        "    host_eligibility: local\n"
        "    versions:\n"
        "      stable:\n"
        "        weights_source: file://not-installed-on-this-machine/blla.safetensors\n"
    )
    return path


# ---------------------------------------------------------------------------
# Claim, fetch, run, callback - end to end, on real weights
# ---------------------------------------------------------------------------
@pytest.mark.ml
def test_a_page_is_claimed_fetched_run_and_reported_end_to_end(
    installed_cli, platform_url, agent
) -> None:
    """The four steps ADR 0003 is built around, across one database and one hop.

    Real **Hub artifact**s: the segmenter here is the PyTorch checkpoint resolved
    through the **Hub cache**, the same one a researcher's laptop resolves.
    """
    ids = agent.new_page()
    job_id = agent.submit_segment(ids)

    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )

    assert run.returncode == 0, run.output
    # Per-page progress, in the order the page went through the loop.
    assert "[1] segment" in run.stdout, run.stdout
    assert "fetched" in run.stdout
    assert "ran in" in run.stdout
    assert "reported done" in run.stdout

    finished = agent.await_job(job_id, status="done")
    assert finished["error"] is None
    assert finished["execution_target"] == "local"
    # The work actually landed: the platform holds lines it did not have before.
    assert len(agent.lines(ids)) > 1


@pytest.mark.ml
def test_the_cli_produces_what_the_same_code_produces_in_process(
    installed_cli, platform_url, agent
) -> None:
    """Local and cloud are the same program, so the CLI must add nothing.

    Compared against `run_model` called here, on the *stored* page bytes rather
    than the uploaded file - the platform normalises uploads to WebP, so the
    bytes the agent fetches are not the bytes that were uploaded, and comparing
    against the wrong ones would be comparing two different pages.
    """
    from inference.contracts.common import InferenceTask
    from inference.jobs.runner import run_model

    ids = agent.new_page()
    job_id = agent.submit_segment(ids)

    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )
    assert run.returncode == 0, run.output
    agent.await_job(job_id, status="done")

    stored = agent.lines(ids)
    in_process = run_model(
        task=InferenceTask.segment,
        registry_model_id="blla-segment",
        registry_tag="stable",
        image_bytes=_stored_page_bytes(agent, ids),
    )

    # Two empty lists are equal, and would prove nothing about either path.
    assert len(in_process.lines) > 1, "the model found nothing, so there is no output to compare"
    assert len(stored) == len(in_process.lines)
    assert [line["points"] for line in stored] == [line.points for line in in_process.lines]


def _stored_page_bytes(agent: Researcher, ids: tuple[str, str]) -> bytes:
    """The page image exactly as the platform holds it.

    Asked for at full width, which is the one request that returns the stored
    object untouched - the same bytes the **signed page image link** serves the
    agent. A thumbnail would be a re-encode, and comparing a model's output on
    two different encodings of a page proves nothing.
    """
    request = urllib.request.Request(f"{agent.platform_url}/media/parts/{ids[1]}", method="GET")
    for name, value in agent.headers.items():
        request.add_header(name, value)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


@pytest.mark.ml
def test_only_one_page_is_ever_in_flight(installed_cli, platform_url, agent) -> None:
    """A batch is N claims, not one claim of N pages (ADR 0002).

    Watched from outside while the loop runs: a page the agent holds is
    `waiting`, so the count of `waiting` jobs is the number in flight. If the
    loop ever claimed ahead, this would see two.
    """
    first = agent.submit_segment(agent.new_page())
    second = agent.submit_segment(agent.new_page())

    run = _start_run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )
    try:
        highest = 0
        while run.process.poll() is None:
            in_flight = sum(
                1 for job_id in (first, second) if agent.job(job_id)["status"] == "waiting"
            )
            highest = max(highest, in_flight)
            assert in_flight <= 1, f"{in_flight} pages were in flight at once"
            time.sleep(0.2)
    finally:
        run.wait()

    assert run.returncode == 0, run.output
    assert highest == 1, "no page was ever observed in flight, so nothing was proved"
    assert agent.await_job(first, status="done")
    assert agent.await_job(second, status="done")
    assert "[2] segment" in run.stdout, run.stdout


# ---------------------------------------------------------------------------
# Every page ends, including the ones that do not finish
# ---------------------------------------------------------------------------
def test_a_page_that_cannot_run_here_is_reported_failed_and_the_loop_continues(
    installed_cli, platform_url, agent, tmp_path
) -> None:
    """A researcher is never left waiting on a page that already died.

    The failure is real rather than injected: this machine is given a
    **Registry** whose segmenter points at a weight file it does not have, which
    is exactly what a half-provisioned laptop looks like.
    """
    first = agent.submit_segment(agent.new_page())
    second = agent.submit_segment(agent.new_page())

    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
        environment={"INFERENCE_REGISTRY_PATH": str(_broken_registry(tmp_path / "registry.yaml"))},
    )

    # The loop did not stop at the first failure, and did not exit non-zero for
    # it either: a page failing here is an outcome, not a reason to give up.
    assert run.returncode == 0, run.output
    assert "[1] segment" in run.stdout
    assert "[2] segment" in run.stdout, "the loop stopped after the first failure"
    assert run.stdout.count("reported failed") == 2, run.stdout

    for job_id in (first, second):
        failed = agent.await_job(job_id, status="failed")
        assert failed["error"], "a failed page must carry its reason"
        assert "BLLA model not found" in failed["error"], failed["error"]


def test_ctrl_c_reports_the_page_in_flight_before_exiting(
    installed_cli, platform_url, agent
) -> None:
    """A considerate shutdown leaves nothing stuck.

    The signal goes to a real process at a real moment - once it has said it is
    holding a page - and the page has to be terminal on the platform afterwards.
    Only a *crash* is allowed to leave a page for the **lease**.
    """
    job_id = agent.submit_segment(agent.new_page())

    run = _start_run(installed_cli, platform_url, agent.home, arguments=("--wait-seconds", "0"))
    try:
        assert run.await_output("[1] segment", timeout=120), run.output
        run.process.send_signal(signal.SIGINT)
        run.wait(timeout=120)
    finally:
        if run.process.poll() is None:  # pragma: no cover - only on an unresponsive CLI
            run.process.kill()

    assert run.returncode == 130, run.output  # 128 + SIGINT, by the shell convention
    assert "reported failed" in run.stdout, run.stdout
    assert "Stopped." in run.stdout

    ended = agent.await_job(job_id, status="failed")
    assert ended["error"], "the interrupted page was reported without a reason"


def test_a_killed_process_leaves_a_page_the_lease_later_releases(
    installed_cli, short_lease_platform_url, tmp_path
) -> None:
    """The other half of the bargain: a crash is covered, and covered differently.

    `SIGKILL` cannot be caught, so nothing reports this page - which is the
    point. The **lease** returns it to the queue as `pending` rather than failing
    it, because a closed lid is not a failed job, and the claim is cleared so any
    agent may take it next.
    """
    home = tmp_path / "nomicous-home"
    researcher = _register(short_lease_platform_url, home)
    _pair(installed_cli, short_lease_platform_url, researcher)
    _announce_capacity(short_lease_platform_url, home)
    researcher.prefer_local()
    job_id = researcher.submit_segment(researcher.new_page())

    run = _start_run(
        installed_cli, short_lease_platform_url, home, arguments=("--wait-seconds", "0")
    )
    assert run.await_output("    fetched", timeout=120), run.output
    run.process.kill()
    run.process.wait(timeout=30)

    # Still held, and held by this agent: nothing reported it.
    held = researcher.job(job_id)
    assert held["status"] == "waiting", held
    assert held["error"] is None

    # The sweep is opportunistic and runs on read paths, so waiting alone proves
    # nothing - the page comes back when somebody asks the platform a question.
    # Asked by *reading* the job rather than by claiming: a claim sweeps and then
    # immediately takes what the sweep just released, which would hide the state
    # this test is about.
    time.sleep(LEASE_SECONDS + 2)
    released = researcher.await_job(job_id, status="pending", timeout=30)
    assert released["error"] is None, "an expired lease must not fail the page"
    # Re-pended, not failed, and the claim cleared with it - so any agent may
    # take the page next, and the killed one could not report on it if it woke.
    assert released["completed_at"] is None


# ---------------------------------------------------------------------------
# An empty queue: the two behaviours one flag chooses between
# ---------------------------------------------------------------------------
def test_an_empty_queue_ends_the_run_when_asked_to(installed_cli, platform_url, agent) -> None:
    """The scripted half: nothing queued, so there is nothing to wait for."""
    started = time.monotonic()
    run = _run(
        installed_cli,
        platform_url,
        agent.home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
    )
    elapsed = time.monotonic() - started

    assert run.returncode == 0, run.output
    assert "Nothing left in the queue." in run.stdout
    assert elapsed < 60, f"it waited {elapsed:.1f}s for a queue it was told to give up on"
    # Being **outdated** is a notice delivered with the work, not a refusal: it
    # is said once and the loop carries on.
    assert NEWER_THAN_THIS_AGENT in run.stdout, run.stdout


def test_an_empty_queue_is_waited_on_by_default(installed_cli, platform_url, agent) -> None:
    """The interactive half: the same binary, one flag apart."""
    run = _start_run(installed_cli, platform_url, agent.home, arguments=("--wait-seconds", "0"))
    try:
        assert run.await_output("Waiting for work", timeout=90), run.output
        time.sleep(3)
        assert run.process.poll() is None, f"it exited on an empty queue:\n{run.output}"

        run.process.send_signal(signal.SIGINT)
        run.wait(timeout=60)
    finally:
        if run.process.poll() is None:  # pragma: no cover - only on an unresponsive CLI
            run.process.kill()

    assert run.returncode == 130, run.output
    assert "Stopped." in run.stdout


# ---------------------------------------------------------------------------
# The two credentials, and the version floor in front of both
# ---------------------------------------------------------------------------
@pytest.mark.ml
def test_a_hosted_worker_runs_the_same_loop_with_a_service_credential(
    installed_cli, platform_url, tmp_path
) -> None:
    """Local and cloud differ by credential and uptime, not by code path (ADR 0003).

    No `device.json` exists in this home directory at all: the only thing making
    this process an **inference agent** is the **service credential** in its
    environment, and the only thing it changes is that the work it takes is
    `cloud` work.
    """
    home = tmp_path / "worker-home"
    home.mkdir(parents=True)
    researcher = _register(platform_url, home)
    worker = {"NOMICOUS_SERVICE_TOKEN": SERVICE_TOKEN, "NOMICOUS_WORKER_NAME": "test-worker"}

    # A hosted worker registers itself by working: this first claim provisions
    # the `cloud` device row that reports cloud **capacity**, without which
    # submission would announce no host rather than create the page.
    status, body = _post(
        f"{platform_url}/device/v1/jobs/claim",
        {"wait_seconds": 0},
        {
            "X-Nomicous-Service-Token": SERVICE_TOKEN,
            "X-Nomicous-Worker-Name": "test-worker",
            "X-Nomicous-Agent-Version": "1.0.0",
        },
    )
    assert status == 200, body

    job_id = researcher.submit_segment(researcher.new_page())
    assert researcher.job(job_id)["execution_target"] == "cloud"
    assert not (home / "device.json").exists()

    run = _run(
        installed_cli,
        platform_url,
        home,
        arguments=("--exit-when-empty", "--wait-seconds", "0"),
        environment=worker,
    )

    assert run.returncode == 0, run.output
    assert "cloud work" in run.stdout, run.stdout
    assert "reported done" in run.stdout, run.output
    assert researcher.await_job(job_id, status="done")["error"] is None


def test_an_agent_below_the_version_floor_is_refused_and_told_what_to_do(
    installed_cli, refusing_platform_url, tmp_path
) -> None:
    """A 426 is an instruction, not a blip to back off from (issue 055).

    Paired first, and paired against this same platform, so the refusal cannot
    be mistaken for a credential problem: the floor is evaluated *before*
    authentication, which is what makes 426 the right status and what makes a
    refused agent stop reporting **capacity** rather than accumulating work it
    may not take.
    """
    home = tmp_path / "old-agent-home"
    researcher = _register(refusing_platform_url, home)
    _pair(installed_cli, refusing_platform_url, researcher)

    run = _run(installed_cli, refusing_platform_url, home, arguments=("--wait-seconds", "0"))

    assert run.returncode != 0, run.output
    assert IMPOSSIBLE_MINIMUM_VERSION in run.stderr, run.stderr
    assert "uv tool upgrade nomicous-inference" in run.stderr
    # It reported and stopped rather than retrying something that cannot succeed.
    assert "[1]" not in run.stdout


def test_an_unpaired_machine_is_told_to_pair_rather_than_left_polling(
    installed_cli, platform_url, tmp_path
) -> None:
    """There is no credential here to claim with, and nothing to wait for."""
    run = _run(installed_cli, platform_url, tmp_path / "empty-home")

    assert run.returncode != 0, run.output
    assert "not paired" in run.stderr.lower()
    assert "nomicous pair" in run.stderr
