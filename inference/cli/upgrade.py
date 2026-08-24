"""Self-upgrade at launch, and only at launch.

A CLI has something a daemon doesn't: a launch moment with no in-flight work.
Before it claims anything, the agent asks the platform what it must be
running (`GET /device/v1/agent/version`), and there are exactly three
answers:

* **below the floor** - install a newer build, then re-exec into it and go on
  to claim. This is the only moment at which the process may replace its own
  code.
* **outdated** - print a notice and claim anyway. Most upgrades aren't
  urgent, and refusing them would turn every release into an outage for
  anyone who hadn't restarted.
* **current** - print nothing and claim.

Never mid-session
-----------------
There is one call site, in `main()`, before the command is dispatched.
Nothing in a claim loop may call back into this module: a process that swaps
its own code while a page is in flight would run one version's model on
another version's downloaded weights, having already told the platform which
version it was. The **lease** would return the page to the queue, but the
wrong output would already have been posted.

A failed upgrade is loud and fatal
----------------------------------
Non-zero exit, a message naming what to do, and no claiming. The failure
worth guarding against is not a crash but a researcher watching a terminal
that looks busy while nothing is being transcribed, so an agent that can't
make itself claimable stops instead of continuing quietly.

Accepted risk, recorded rather than mitigated
---------------------------------------------
Auto-upgrade executes newly fetched code without asking. A compromised
`nomikos-inference` on the index this machine installs from therefore
reaches every researcher's laptop at its next launch, with no human in the
loop. Mitigable by pinning to published hashes; not eliminable, since the
point of the feature is to install something nobody has approved yet. Two
things narrow the risk rather than close it: the platform names neither a
command to run nor *which* package to install (the name is pinned to
`DISTRIBUTION_NAME` here, and a mismatch is fatal via
`_reject_unexpected_target`, so the only thing a compromised platform can
choose is the *version* of this one distribution); and the installer is
whichever one already owns this environment, so the index is the one the
researcher configured, not one the platform picked.

What's left is the index: a compromised `nomikos-inference` there still
reaches every laptop. What's no longer possible is the platform pointing this
process at some *other* distribution, which would be code execution by a
different route - an sdist runs its build hooks, and a wheel declaring a
`nomikos` console script replaces the entry point `_re_exec` is about to run.

A notice telling researchers to upgrade themselves was considered and
rejected: it's safer and it's ignorable, and stale agents are exactly the
population that ignores notices (ADR 0002).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys

from inference.cli import console as ui
from inference.cli.api import (
    AgentFloor,
    InsecurePlatformURL,
    PlatformClient,
    PlatformError,
    default_platform_url,
)
from inference.cli.credentials import CredentialError, load_credential
from inference.cli.version import DISTRIBUTION_NAME, SOURCE_CHECKOUT_VERSION, installed_version

MAX_FLOOR_VERSION_LENGTH = 32
"""Mirrors ``MAX_AGENT_VERSION_LENGTH`` on the platform side. A floor longer
than the column that records a version isn't a version."""

_FLOOR_VERSION_PATTERN = re.compile(
    r"""
    ^
    (?:0|[1-9][0-9]*)
    \.(?:0|[1-9][0-9]*)
    \.(?:0|[1-9][0-9]*)
    (?:[-.]?(?:a|b|rc|alpha|beta|dev)\.?(?:0|[1-9][0-9]*)?)?
    (?:\+[0-9a-zA-Z.]+)?
    $
    """,
    re.VERBOSE,
)
"""The same grammar ``backend/ml/domain/agent_version.py`` accepts, restated
on this side of the wire.

Deliberately a copy rather than an import: this package ships to a
researcher's laptop and the platform's domain module doesn't travel with it.
The value it guards is interpolated into an installer argument, so the client
has to be able to reject a floor without asking anyone. Narrower than PEP 440
on purpose, for the same reason the platform's is: anything outside it is
refused rather than guessed at."""

UPGRADED_FROM_ENV = "NOMIKOS_UPGRADED_FROM"
"""Set on the process an upgrade re-execs into, holding the version it came
from. Its presence stops a second upgrade: if the new build is *still* below
the floor (an index with nothing newer, a floor above the newest release),
upgrading again would fetch the same wheel and exec again for as long as the
machine is powered on. One attempt, then a fatal error naming both numbers."""

INSTALL_TIMEOUT_SECONDS = 600.0
"""A wheel and its closure over a slow connection. Long enough not to fail a
researcher on hotel wifi, short enough that a hung index isn't indefinite."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-url",
        metavar="URL",
        help=(
            "Platform to ask for the version floor. Defaults to the one this machine "
            "is paired with, then $NOMIKOS_API_URL, then the hosted API."
        ),
    )


def run(args: argparse.Namespace) -> int:
    """`nomikos upgrade` - the launch check, run on demand.

    Deliberately the same code path and output as the check that runs before
    claiming, down to printing nothing when this agent is current. A
    subcommand that reported *more* than the launch check would be a second
    implementation of the one thing here that must not have two.
    """
    return check_before_claiming(args)


def check_before_claiming(args: argparse.Namespace) -> int:
    """Bring this agent up to the platform's floor, or refuse to claim.

    Returns `EXIT_OK` when claiming may begin. Doesn't return at all when an
    upgrade succeeded: the process is replaced by the new build, which runs
    this same check and finds itself current.
    """
    console = ui.out()
    errors = ui.err()
    agent_version = installed_version()

    try:
        floor = _ask(args, agent_version)
    except InsecurePlatformURL as exc:
        # Unlike an unreachable platform, the next command wouldn't fail with
        # a better message here - it would fail with this same one. Saying it
        # once, here, keeps it from reading as a network blip.
        errors.print(f"[red]{exc}[/red]")
        return ui.EXIT_FAILED
    except PlatformError as exc:
        # Not a reason to stop. A researcher on a train hasn't been refused,
        # and whatever they were about to do will fail on its own terms with
        # a better message than this one could give.
        _verbatim(errors, f"{exc} Continuing as {agent_version}.")
        return ui.EXIT_OK

    if not floor.refused:
        if floor.outdated:
            console.print(
                f"[yellow]{floor.package} {agent_version} is behind "
                f"{floor.latest_version}.[/yellow] Work is still being handed over; "
                "upgrade when convenient:"
            )
            # A hint for a human to read, and the one string here the platform
            # supplies whole. Printed, never executed.
            _verbatim(
                console,
                f"  {floor.upgrade_command or f'uv tool upgrade {floor.package}'}",
                style="",
            )
            console.print()
        return ui.EXIT_OK

    return _upgrade_or_stop(console, errors, floor, agent_version)


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------
def _ask(args: argparse.Namespace, agent_version: str) -> AgentFloor:
    return PlatformClient(platform_url_for(args)).read_agent_floor(agent_version=agent_version)


def platform_url_for(args: argparse.Namespace) -> str:
    """Which platform decides. Explicit flag, then the paired one, then default.

    The credential comes second rather than first because `--api-url` is how
    a researcher points a machine at a staging platform, and it would be
    strange for the floor to come from one platform while the **claim** went
    to another. An unreadable credential isn't this check's problem to
    report (the command that needs it will say so), so it falls through to
    the default here.
    """
    explicit = getattr(args, "api_url", None)
    if explicit:
        return str(explicit).rstrip("/")
    try:
        credential = load_credential()
    except CredentialError:
        credential = None
    if credential is not None and credential.platform_url:
        return credential.platform_url.rstrip("/")
    return default_platform_url()


# ---------------------------------------------------------------------------
# Upgrading
# ---------------------------------------------------------------------------
def _verbatim(console, raw: str, *, style: str = "dim") -> None:
    """Print something this program did not write, exactly as it is.

    Installer output, filesystem paths, and the platform's own sentences all
    end up here, and none of them is trusted to be Rich markup: `pip` alone
    emits lines like `[notice] A new release of pip is available`, which as
    markup is an unknown style and an exception rather than a line of text.
    Never wrapped either, since a command a researcher is meant to re-run
    must arrive in one piece.
    """
    text = ui.value(raw)
    if style:
        text.stylize(style)
    console.print(text, soft_wrap=True)


def _reject_unexpected_target(errors, floor: AgentFloor) -> bool:
    """True when the floor names something this agent must not install.

    Two fields off the wire reach an installer argument, and both are
    checked here before anything else in the upgrade path looks at them.

    ``package`` is pinned to `DISTRIBUTION_NAME`. Upgrading *this* agent
    means installing a newer build of the distribution it already is; any
    other name is not an upgrade, and running an installer on it would hand
    the platform the choice of what executes on the researcher's machine.
    There's no legitimate reason for the two to differ, including for a
    self-hosted platform: the agent it's talking to is `nomikos-inference`
    whatever the platform is called.

    ``minimum_version`` is held to the platform's own version grammar. It's
    interpolated into ``package>=version``, and a requirement specifier is a
    small language of its own - a floor of ``0 --index-url http://…`` is not
    a version, and neither is anything else the grammar doesn't admit.
    """
    if floor.package != DISTRIBUTION_NAME:
        errors.print(
            f"[red]The platform asked this agent to install a different package.[/red] "
            f"Only {DISTRIBUTION_NAME} can be upgraded here. Nothing was installed "
            f"and nothing was claimed."
        )
        errors.print("It named:")
        _verbatim(errors, f"  {floor.package}", style="")
        return True

    if (
        len(floor.minimum_version) > MAX_FLOOR_VERSION_LENGTH
        or _FLOOR_VERSION_PATTERN.match(floor.minimum_version) is None
    ):
        errors.print(
            "[red]The platform named a floor that is not a version.[/red] "
            "Nothing was installed and nothing was claimed."
        )
        errors.print("It named:")
        _verbatim(errors, f"  {floor.minimum_version}", style="")
        return True

    return False


def _upgrade_or_stop(console, errors, floor: AgentFloor, agent_version: str) -> int:
    # The platform's own sentence, printed rather than reworded so a
    # researcher and the server logs say the same thing, and printed inertly
    # since it arrived over the wire.
    _verbatim(console, floor.message, style="yellow")

    # Boundary check, before any branch below quotes these fields or builds a
    # requirement out of them: past here, `floor.package` is known to be this
    # distribution and `floor.minimum_version` is known to be a version.
    if _reject_unexpected_target(errors, floor):
        return ui.EXIT_FAILED

    if agent_version == SOURCE_CHECKOUT_VERSION:
        # No distribution here to replace, and installing one over a
        # checkout would leave two copies of the CLI on the path.
        errors.print(
            f"[red]This is a source checkout, not an installed {DISTRIBUTION_NAME}.[/red] "
            f"Nothing can be upgraded in place; install {floor.package} "
            f"{floor.minimum_version} or newer to claim work."
        )
        return ui.EXIT_FAILED

    previous_attempt = os.environ.get(UPGRADED_FROM_ENV)
    if previous_attempt:
        errors.print(
            f"[red]Upgrading {floor.package} did not fix this.[/red] "
            f"This agent was {previous_attempt}, is now {agent_version}, and "
            f"{floor.minimum_version} or newer is required. Nothing was claimed."
        )
        errors.print(
            f"The index this machine installs from may have nothing newer than "
            f"{agent_version} on it."
        )
        return ui.EXIT_FAILED

    requirement = f"{floor.package}>={floor.minimum_version}"
    command = _installer_command(requirement)
    if command is None:
        errors.print(
            f"[red]No installer is available to upgrade {floor.package}.[/red] "
            f"Neither pip nor uv can be found from {sys.executable}, so this agent "
            f"cannot bring itself to {floor.minimum_version}. Nothing was claimed."
        )
        return ui.EXIT_FAILED

    console.print(f"Upgrading {requirement} ...")
    # Which installer, printed before it runs: the one fact support needs
    # when an upgrade lands somewhere unexpected, and not knowable afterwards
    # from anything the CLI leaves behind.
    _verbatim(console, f"  {' '.join(command)}")
    completed = _install(command)
    if completed.returncode != 0:
        errors.print(f"[red]Upgrading {floor.package} failed.[/red] Nothing was claimed.")
        _verbatim(errors, f"  {' '.join(command)}")
        output = (completed.stdout or "") + (completed.stderr or "")
        if output.strip():
            _verbatim(errors, output.strip(), style="")
        errors.print("Upgrade this agent yourself and start it again:")
        _verbatim(
            errors, f"  {floor.upgrade_command or f'uv tool upgrade {floor.package}'}", style=""
        )
        return ui.EXIT_FAILED

    console.print(f"Upgraded {floor.package} from {agent_version}. Restarting.")
    _re_exec(agent_version)
    # `_re_exec` replaces this process; reaching here means it could not.
    errors.print(
        f"[red]{floor.package} was upgraded but this agent could not restart into it.[/red] "
        "Nothing was claimed. Start it again to pick the new version up."
    )
    return ui.EXIT_FAILED


def _installer_command(requirement: str) -> list[str] | None:
    """Whichever installer already owns this environment, asked for one package.

    pip first: if this interpreter has it, it's how the environment was built
    and it installs exactly where this code is running from. A `uv tool
    install` environment has no pip in it at all, which is why uv is the
    fallback and not an afterthought - it's the installer the documented
    install path uses.

    `--python sys.executable` on the uv branch, because a `uv` on `$PATH` has
    no other reason to touch this virtual environment and would otherwise
    resolve one from the working directory.
    """
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--upgrade", requirement]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, requirement]
    return None


def _install(command: list[str]) -> subprocess.CompletedProcess:
    try:
        # S603: `command` is built by `_install_command` as a fixed argv list,
        # no shell, no interpolation. Its only variable part is `requirement`,
        # whose package half is pinned to `DISTRIBUTION_NAME` and whose
        # version half must match `_FLOOR_VERSION_PATTERN` under
        # `MAX_FLOOR_VERSION_LENGTH` (see `_refuses_floor`), so the platform
        # can't smuggle a second argument such as `--index-url` through it.
        return subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command,
            returncode=1,
            stdout="",
            stderr=f"The installer did not finish within {INSTALL_TIMEOUT_SECONDS:g}s.",
        )
    except OSError as exc:  # pragma: no cover - the executable vanished mid-run
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr=str(exc))


def _re_exec(previous_version: str) -> None:
    """Restart into the build that was just installed, with the same arguments.

    `exec` rather than a child process, so there's one agent on this machine
    and not two: the platform counts **capacity** by device, and a parent
    waiting on a child would be a second process holding the same credential.

    A running interpreter can't pick up a package it's already imported, so
    the new code only exists after the image is replaced. Everything the new
    process needs to know about the old one travels in the environment.
    """
    environment = dict(os.environ)
    environment[UPGRADED_FROM_ENV] = previous_version

    script = sys.argv[0]
    if script and os.path.isfile(script) and os.access(script, os.X_OK):
        # The console script the researcher actually ran. It points at this
        # same interpreter, so it starts the build just installed into it.
        arguments = [script, *sys.argv[1:]]
    else:
        # Started as `python -m inference.cli`, or through something that
        # left no usable `argv[0]`. The module path resolves to the new code
        # just as well.
        script = sys.executable
        arguments = [sys.executable, "-m", "inference.cli", *sys.argv[1:]]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        # S606: no shell and no `$PATH` lookup - `script` is either this
        # process's own `argv[0]`, checked to be an executable file above, or
        # `sys.executable`. `arguments` is this process's own argv.
        os.execve(script, arguments, environment)  # noqa: S606
    except OSError:
        return
