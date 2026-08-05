"""`nomicous upgrade` - the launch check - against a real index and a real platform.

Everything here is live, and the package index is the part that had to be. The
agent is a real `nomicous` console script installed from a real wheel; the index
it upgrades from is a PEP 503 tree of real wheels served over real HTTP by
`python -m http.server`; the installer is the real `pip` or the real `uv`,
resolving over the network to that index; and the platform is a real uvicorn
process serving the real `create_app()` against real Postgres. The re-exec is a
real `execve`, observed by reading the version off the process that came out the
other side.

That matters because every claim this issue makes is about what happens *between*
processes. "Upgrades, re-execs, and then claims" is three processes and two
network services; a test that patched `subprocess.run` would be asserting that
this module knows what it wrote, and would have proved nothing about whether a
researcher's laptop comes back up on the new build.

Two things about the wheels are deliberately unlike production, and neither is a
substitution for anything under test. They are built from the repository's own
`inference/` tree with a generated `pyproject.toml`, because two versions of one
package cannot both be the checked-in version - the *code* in them is this
repository's. And they declare no dependencies, so the resolver never leaves the
local index; whether the published closure resolves is
`test_published_package.py`'s question, and answering it here would download
Torch twice per test.

Uses its own database (`kalamos_058_upgrade`) for the same reason
`test_cli_pairing.py` does: servers are held open across the module.
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
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT

pytestmark = pytest.mark.integration

DATABASE = "kalamos_058_upgrade"
POSTGRES_DSN = "postgresql://postgres:dev@localhost:5433"
APP_ORIGIN = "https://app.nomicous.test"

SERVER_START_TIMEOUT_SECONDS = 60.0
CLI_TIMEOUT_SECONDS = 300.0

PACKAGE = "nomicous-inference"

# The two builds on the index. `0.9.0` beats `0.1.0` by arithmetic and not by
# alphabet, which is the ordering the platform's floor is careful about.
AGENT_OLD = "0.1.0"
AGENT_NEW = "0.9.0"
# Above everything the index has, so a floor set here is one no upgrade can
# reach - which is how the failure path is provoked without breaking the index.
BEYOND_INDEX = "1.5.0"

FLOOR_PATH = "/device/v1/agent/version"

_PYPROJECT = """\
[project]
name = "nomicous-inference"
version = "{version}"
description = "Agent build {version}, for the self-upgrade tests"
requires-python = ">=3.11,<3.13"
dependencies = []

[project.scripts]
nomicous = "inference.cli:main"

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["inference"]
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        pytest.skip("uv is required to build the wheels this module upgrades between")
    return executable


# ---------------------------------------------------------------------------
# A real package index: two wheels, PEP 503 layout, served over HTTP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def local_index(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build both agent builds and serve them as a simple index.

    Session-scoped because it is two wheel builds and one HTTP server, and every
    test in the module resolves against the same index.
    """
    uv = _uv()
    workspace = tmp_path_factory.mktemp("index")
    source = workspace / "package"
    shutil.copytree(
        REPO_ROOT / "inference",
        source / "inference",
        # The loopback HTTP surfaces are excluded from the published wheel too,
        # and nothing in the CLI imports them.
        ignore=shutil.ignore_patterns("api", "helper", "__pycache__", "*.pyc"),
    )

    root = workspace / "index"
    project = root / "simple" / PACKAGE
    project.mkdir(parents=True)

    for version in (AGENT_OLD, AGENT_NEW):
        (source / "pyproject.toml").write_text(_PYPROJECT.format(version=version))
        built = subprocess.run(
            [uv, "build", "--wheel", "-o", str(project), str(source)],
            capture_output=True,
            text=True,
        )
        assert built.returncode == 0, built.stderr

    wheels = sorted(project.glob("*.whl"))
    assert len(wheels) == 2, wheels
    links = "\n".join(f'<a href="{wheel.name}">{wheel.name}</a><br/>' for wheel in wheels)
    (project / "index.html").write_text(f"<!DOCTYPE html><html><body>{links}</body></html>\n")
    (root / "simple" / "index.html").write_text(
        f'<!DOCTYPE html><html><body><a href="{PACKAGE}/">{PACKAGE}</a></body></html>\n'
    )

    port = _free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    index_url = f"http://127.0.0.1:{port}/simple/"
    try:
        _wait_for(server, f"{index_url}{PACKAGE}/", "the local package index")
        yield index_url
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            server.kill()


def _wait_for(process: subprocess.Popen, url: str, what: str) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"{what} exited before serving")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.2)
    process.terminate()
    raise AssertionError(f"{what} did not answer {url} in time")


# ---------------------------------------------------------------------------
# The platform: real app, real Postgres, one server per version policy
# ---------------------------------------------------------------------------
def _psql(sql: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "nomicous-db-1", "psql", "-U", "postgres", "-c", sql],
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def migrated_database() -> str:
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


@dataclass
class Platform:
    """One running platform, and the log that says what was asked of it."""

    base_url: str
    log_path: Path

    def requests_to(self, path: str) -> int:
        """How many times this path has been requested, off the access log.

        The uvicorn access log is the only place a *second* process's HTTP
        behaviour is recorded, which is what "the launch check runs once" needs
        evidence of - the CLI cannot be asked, and counting inside it would only
        count what this module already believes.
        """
        return sum(1 for line in self.log_path.read_text().splitlines() if f" {path} " in line)


@pytest.fixture(scope="session")
def platform_at(migrated_database: str, tmp_path_factory: pytest.TempPathFactory):
    """Start (and reuse) a platform with a given version floor and latest release.

    One server per policy rather than one server reconfigured, because
    `DeviceSettings` is cached for the life of the process - which is the point
    of `@settings_cache`, and not something a test should be reaching around.
    """
    servers: dict[tuple[str, str], tuple[subprocess.Popen, Platform]] = {}
    logs = tmp_path_factory.mktemp("platforms")

    def start(*, minimum: str, latest: str) -> Platform:
        key = (minimum, latest)
        if key in servers:
            return servers[key][1]

        port = _free_port()
        log_path = logs / f"server-{minimum}-{latest}.log"
        environment = dict(os.environ)
        environment.update(
            {
                "MIGRATOR_DATABASE_URL": migrated_database,
                "SYNC_DATABASE_URL": migrated_database,
                "DATABASE_URL": migrated_database.replace("postgresql://", "postgresql+asyncpg://"),
                "JWT_SECRET": "test-secret-not-for-production-at-least-32-bytes",
                "DEVICE_TOKEN_HMAC_SECRET": "test-device-token-hmac-secret-not-for-production",
                "DEVICE_PAIRING_ENABLED": "true",
                "DEVICE_PAIRING_APP_ORIGIN": APP_ORIGIN,
                "AUTH_RATE_LIMIT_REQUESTS": "1000",
                "JOB_WORKER_ENABLED": "false",
                "ENVIRONMENT": "development",
                "INFERENCE_AGENT_MIN_VERSION": minimum,
                "INFERENCE_AGENT_LATEST_VERSION": latest,
                "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "nomicous"), str(REPO_ROOT)]),
                "INFERENCE_REGISTRY_PATH": str(REPO_ROOT / "inference" / "registry.yaml"),
            }
        )
        environment.pop("INFERENCE_WORKER_SERVICE_TOKEN", None)

        log_file = log_path.open("w")
        # `info` rather than `warning`: the access log is the evidence this
        # module reads back. Output goes to a file, never a pipe - SQLAlchemy
        # echo is on outside production and would fill a pipe buffer and block
        # the server mid-request.
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
                "info",
            ],
            cwd=REPO_ROOT / "nomicous",
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        platform = Platform(base_url=f"http://127.0.0.1:{port}", log_path=log_path)
        servers[key] = (server, platform)
        _wait_for(server, f"{platform.base_url}/health", f"the platform (floor {minimum})")
        return platform

    try:
        yield start
    finally:
        for server, _ in servers.values():
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover
                server.kill()


# ---------------------------------------------------------------------------
# The agent: a real console script in its own environment
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    """One installed agent, and how to run it."""

    executable: Path
    python: Path
    home: Path
    workspace: Path
    environment: dict[str, str]

    def run(
        self, *arguments: str, extra: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        environment = dict(self.environment)
        environment.update(extra or {})
        return subprocess.run(
            [str(self.executable), *arguments],
            env=environment,
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )

    def installed_version(self) -> str:
        """What is on disk now, read the way the floor reads it."""
        completed = subprocess.run(
            [
                str(self.python),
                "-c",
                "from importlib.metadata import version; print(version('nomicous-inference'))",
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()


@pytest.fixture
def agent_at(local_index: str, tmp_path: Path):
    """Install one agent build into an empty environment, and hand it back.

    `with_pip` picks which installer the agent will find. Both branches are real
    install paths: `uv tool install` leaves no pip in the environment at all, and
    `pip install` leaves one. The CLI has to bring itself forward either way.
    """
    uv = _uv()
    counter = [0]

    def install(version: str, *, with_pip: bool = False) -> Agent:
        counter[0] += 1
        workspace = tmp_path / f"agent-{counter[0]}"
        venv = workspace / "venv"
        home = workspace / "nomicous-home"
        workspace.mkdir(parents=True)

        create = [
            uv,
            "venv",
            str(venv),
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}",
        ]
        if with_pip:
            create.append("--seed")
        subprocess.run(create, check=True, capture_output=True, text=True)

        scripts = venv / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / "python"
        installed = subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--index-url",
                local_index,
                f"{PACKAGE}=={version}",
            ],
            capture_output=True,
            text=True,
        )
        assert installed.returncode == 0, installed.stderr
        # `rich` is a real dependency of the real package; the test wheels drop
        # their dependency list so the upgrade resolves entirely against the
        # local index, so it is installed here instead.
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), "rich"],
            check=True,
            capture_output=True,
            text=True,
        )

        environment = dict(os.environ)
        for inherited in ("NOMICOUS_API_URL", "NOMICOUS_UPGRADED_FROM", "VIRTUAL_ENV"):
            environment.pop(inherited, None)
        environment.update(
            {
                "NOMICOUS_HOME": str(home),
                "PYTHONUNBUFFERED": "1",
                # The index the *installer* reaches, exactly as a researcher's
                # own configuration would reach it. The CLI has no index flag:
                # it upgrades with whatever installer already owns the
                # environment, from wherever that installer is pointed.
                "UV_INDEX_URL": local_index,
                "PIP_INDEX_URL": local_index,
                "PIP_TRUSTED_HOST": "127.0.0.1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                # Rich wraps at the console width, and with no terminal that is
                # 80 columns - narrow enough to break a sentence this module
                # asserts on across two lines. A width, not a mock: the CLI
                # would read the same value from a real terminal.
                "COLUMNS": "200",
            }
        )

        executable = scripts / ("nomicous.exe" if os.name == "nt" else "nomicous")
        assert executable.is_file(), "the wheel did not install a `nomicous` console script"
        return Agent(
            executable=executable,
            python=python,
            home=home,
            workspace=workspace,
            environment=environment,
        )

    return install


# ---------------------------------------------------------------------------
# A real device credential, so "and then claims" can mean a real claim
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
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def pair_a_device(platform: Platform, *, agent_version: str) -> str:
    """Run the pairing protocol over HTTP and return the **device token**.

    The CLI's own `pair` is exercised in `test_cli_pairing.py`; what this module
    needs is a credential that really authenticates, so that the claim it makes
    after an upgrade is a real claim rather than an unauthenticated probe.
    """
    suffix = uuid.uuid4().hex[:8]
    status, account = _post(
        f"{platform.base_url}/auth/register",
        {
            "email": f"upgrade-{suffix}@test.kalamos",
            "username": f"upgrade_{suffix}",
            "password": "test-pass-123",
        },
    )
    assert status == 201, account
    headers = {"Authorization": f"Bearer {account['access_token']}"}

    status, started = _post(
        f"{platform.base_url}/device/v1/pairings",
        {
            "device_name": f"agent-{suffix}",
            "platform": "test-arm64",
            "helper_version": agent_version,
            "capabilities": {"runtime": "torch"},
        },
    )
    assert status == 201, started

    token = started["verification_url"].split("#", 1)[1]
    status, looked_up = _post(
        f"{platform.base_url}/devices/pairings/lookup", {"verification_token": token}, headers
    )
    assert status == 200, looked_up
    status, approved = _post(
        f"{platform.base_url}/devices/pairings/{looked_up['pairing_id']}/approve",
        {"verification_token": token},
        headers,
    )
    assert status in (200, 204), approved

    status, collected = _post(
        f"{platform.base_url}/device/v1/pairings/token",
        {"pairing_id": started["pairing_id"], "device_code": started["device_code"]},
    )
    assert status == 200 and collected.get("device_token"), collected
    return collected["device_token"]


def claim(platform: Platform, *, device_token: str, agent_version: str) -> tuple[int, dict]:
    """One real **claim**, stating a version. The queue is empty, so it is a 200
    with no page - which is the normal state of a healthy platform and is still
    a claim the platform agreed to serve."""
    return _post(
        f"{platform.base_url}/device/v1/jobs/claim",
        {"wait_seconds": 0},
        {
            "X-Nomicous-Device-Token": device_token,
            "X-Nomicous-Agent-Version": agent_version,
        },
    )


# ---------------------------------------------------------------------------
# Below the floor: upgrade, re-exec, then claim
# ---------------------------------------------------------------------------
def test_an_agent_below_the_floor_upgrades_re_execs_and_then_claims(
    agent_at, platform_at, local_index
) -> None:
    """The whole point of the issue, end to end across three processes.

    Before: the platform refuses this build a claim, with 426. The CLI upgrades
    itself from a real index, `execve`s into what it installed, and the version
    that comes out the other side is the new one. After: the same credential
    claiming with the new version is served.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD)
    device_token = pair_a_device(platform, agent_version=AGENT_OLD)

    refused, body = claim(platform, device_token=device_token, agent_version=AGENT_OLD)
    assert refused == 426, body
    assert body["error"]["code"] == "AGENT_VERSION_UNSUPPORTED"

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Upgrading" in completed.stdout, completed.stdout
    assert "Restarting" in completed.stdout, completed.stdout
    assert agent.installed_version() == AGENT_NEW

    served, body = claim(platform, device_token=device_token, agent_version=AGENT_NEW)
    assert served == 200, body
    assert body["agent"]["agent_version"] == AGENT_NEW
    assert body["agent"]["outdated"] is False


def test_the_upgraded_process_is_the_one_that_carries_on(agent_at, platform_at) -> None:
    """`execve`, not a child process: one agent on this machine, not two.

    The re-exec re-runs the same argument vector, so the process that continues
    past the check is the new build running the command the researcher asked
    for. Proved by handing it `--version`, which the new build answers with its
    own number - a number the process that was launched could not have printed.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD)

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    reported = agent.run("--version")
    assert reported.stdout.strip() == f"nomicous {AGENT_NEW}", reported.stdout


# ---------------------------------------------------------------------------
# Merely outdated: a notice, and no upgrade
# ---------------------------------------------------------------------------
def test_an_agent_that_is_merely_outdated_is_told_and_left_alone(agent_at, platform_at) -> None:
    """Outdated is deliberately not the same state as below the floor.

    Most upgrades are not urgent, and refusing them would make every release an
    outage for anyone who had not restarted. So: a notice, the same build still
    on disk afterwards, and a claim the platform serves.
    """
    platform = platform_at(minimum=AGENT_OLD, latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD)
    device_token = pair_a_device(platform, agent_version=AGENT_OLD)

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert AGENT_NEW in completed.stdout, completed.stdout
    assert "Upgrading" not in completed.stdout, completed.stdout
    assert agent.installed_version() == AGENT_OLD, "an outdated agent upgraded itself"

    served, body = claim(platform, device_token=device_token, agent_version=AGENT_OLD)
    assert served == 200, body
    assert body["agent"]["outdated"] is True


# ---------------------------------------------------------------------------
# Current: nothing at all
# ---------------------------------------------------------------------------
def test_an_agent_at_the_current_version_prints_nothing(agent_at, platform_at) -> None:
    """A launch check that announced itself every time would train researchers
    to ignore the one launch where it had something to say."""
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_NEW)

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "", completed.stdout
    assert completed.stderr.strip() == "", completed.stderr
    assert agent.installed_version() == AGENT_NEW


# ---------------------------------------------------------------------------
# Failure is loud, fatal, and claims nothing
# ---------------------------------------------------------------------------
def test_a_failed_upgrade_exits_non_zero_and_claims_nothing(agent_at, platform_at) -> None:
    """A floor no build on the index can reach: the installer really fails.

    The failure mode being engineered against is a researcher watching a
    terminal that looks busy while nothing is being transcribed - so an agent
    that cannot make itself claimable stops, says why, and exits non-zero.
    """
    platform = platform_at(minimum=BEYOND_INDEX, latest=BEYOND_INDEX)
    agent = agent_at(AGENT_OLD)
    device_token = pair_a_device(platform, agent_version=AGENT_OLD)

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode != 0, completed.stdout
    assert "failed" in completed.stderr.lower(), completed.stderr
    assert "Nothing was claimed" in completed.stderr, completed.stderr
    # It names the recovery, and it is still the build it started as.
    assert PACKAGE in completed.stderr
    assert agent.installed_version() == AGENT_OLD

    refused, body = claim(platform, device_token=device_token, agent_version=AGENT_OLD)
    assert refused == 426, body


def test_an_upgrade_that_did_not_take_stops_instead_of_re_execing_forever(
    agent_at, platform_at
) -> None:
    """The loop guard, from the far side of a re-exec.

    A process that upgraded, restarted, and finds itself *still* below the floor
    would fetch the same wheel and exec again for as long as the machine is
    powered on. One attempt per launch: the environment carries the version it
    came from, and finding it set turns the second refusal into a fatal error
    naming both numbers.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD)

    completed = agent.run(
        "upgrade", "--api-url", platform.base_url, extra={"NOMICOUS_UPGRADED_FROM": "0.0.9"}
    )

    assert completed.returncode != 0, completed.stdout
    assert "did not fix this" in completed.stderr, completed.stderr
    assert "0.0.9" in completed.stderr and AGENT_OLD in completed.stderr
    # No second attempt was made: nothing was installed, nothing was restarted.
    assert "Upgrading" not in completed.stdout, completed.stdout
    assert agent.installed_version() == AGENT_OLD


def test_a_source_checkout_is_told_to_install_rather_than_upgraded(platform_at, tmp_path) -> None:
    """There is no distribution here to replace.

    A source checkout reports `0+unknown`, which the floor refuses on the same
    terms as a version that is too old. Installing a wheel over the checkout
    would leave two copies of the CLI on the path, so this stops and says what
    would actually fix it.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "NOMICOUS_HOME": str(tmp_path / "home"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    environment.pop("NOMICOUS_API_URL", None)
    environment.pop("NOMICOUS_UPGRADED_FROM", None)

    completed = subprocess.run(
        [sys.executable, "-m", "inference.cli", "upgrade", "--api-url", platform.base_url],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
    )

    assert completed.returncode != 0, completed.stdout
    assert "source checkout" in completed.stderr, completed.stderr
    assert "install" in completed.stderr.lower()


def test_a_platform_that_cannot_be_reached_does_not_stop_the_agent(agent_at) -> None:
    """A researcher on a train has not been refused, and must not be told so.

    Whatever the agent was about to do will fail on its own terms with a better
    message than this check could give, so an unanswerable floor is a dim line
    and not an exit status.
    """
    agent = agent_at(AGENT_OLD)
    unreachable = f"http://127.0.0.1:{_free_port()}"

    completed = agent.run("upgrade", "--api-url", unreachable)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Continuing as" in completed.stderr, completed.stderr
    assert agent.installed_version() == AGENT_OLD


# ---------------------------------------------------------------------------
# Both installers, because both are real install paths
# ---------------------------------------------------------------------------
def test_an_environment_with_pip_upgrades_with_pip(agent_at, platform_at) -> None:
    """`pip install nomicous-inference` leaves a pip behind, and it is the
    installer that owns that environment."""
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD, with_pip=True)

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "-m pip install" in completed.stdout, completed.stdout
    assert agent.installed_version() == AGENT_NEW


def test_an_environment_with_no_pip_upgrades_with_uv(agent_at, platform_at) -> None:
    """`uv tool install nomicous-inference` - the documented install path -
    leaves no pip in the environment at all."""
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD)

    has_pip = subprocess.run(
        [str(agent.python), "-c", "import pip"], capture_output=True, text=True
    )
    assert has_pip.returncode != 0, "this environment was supposed to have no pip"

    completed = agent.run("upgrade", "--api-url", platform.base_url)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "uv pip install" in completed.stdout, completed.stdout
    assert agent.installed_version() == AGENT_NEW


# ---------------------------------------------------------------------------
# Asked once, at launch, and never by a command that does not claim
# ---------------------------------------------------------------------------
def test_the_floor_is_asked_once_per_launch(agent_at, platform_at) -> None:
    """One question per process, answered before anything else happens.

    Counted off the platform's own access log, because the number that matters
    is how many times the *other* process asked. A launch check that re-asked
    would be a check that could fire mid-batch.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_NEW)

    before = platform.requests_to(FLOOR_PATH)
    completed = agent.run("upgrade", "--api-url", platform.base_url)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    assert platform.requests_to(FLOOR_PATH) - before == 1


def test_commands_that_do_not_claim_never_ask_the_floor(agent_at, platform_at) -> None:
    """`version` reports what this build is without asking the platform
    anything, and `pair` must work on a machine that cannot claim yet - a floor
    it is not about to test has no business blocking it."""
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)
    agent = agent_at(AGENT_OLD)

    before = platform.requests_to(FLOOR_PATH)
    reported = agent.run("version", extra={"NOMICOUS_API_URL": platform.base_url})
    helped = agent.run("--help")

    assert reported.returncode == 0, reported.stderr
    assert helped.returncode == 0, helped.stderr
    assert platform.requests_to(FLOOR_PATH) == before


# ---------------------------------------------------------------------------
# The endpoint itself
# ---------------------------------------------------------------------------
def test_asking_for_the_floor_takes_nothing_from_the_queue(platform_at) -> None:
    """The reason this is not the claim endpoint.

    An agent that had to claim in order to learn it was stale would be holding a
    page at the exact moment it replaced its own code. This answers the same
    verdict with no credential, no session, and no page.
    """
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)

    request = urllib.request.Request(f"{platform.base_url}{FLOOR_PATH}", method="GET")
    request.add_header("X-Nomicous-Agent-Version", AGENT_NEW)
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        notice = json.loads(response.read())

    assert notice["minimum_version"] == "0.5.0"
    assert notice["latest_version"] == AGENT_NEW
    assert notice["package"] == PACKAGE
    assert notice["outdated"] is False
    assert "page" not in notice


def test_an_agent_that_states_no_version_is_refused_by_the_floor_endpoint(platform_at) -> None:
    """Missing is refused on the same terms as too old - the launch check has to
    see the same verdict the claim path would have given it."""
    platform = platform_at(minimum="0.5.0", latest=AGENT_NEW)

    status, body = _get_json(f"{platform.base_url}{FLOOR_PATH}")

    assert status == 426, body
    assert body["error"]["code"] == "AGENT_VERSION_UNSUPPORTED"
    assert body["error"]["reason"] == "missing"
    assert body["error"]["retryable"] is False


def _get_json(url: str) -> tuple[int, dict]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")
