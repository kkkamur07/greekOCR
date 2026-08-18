"""The upgrade path may install exactly one distribution, at a stated version.

`_upgrade_or_stop` runs an installer on a requirement built from two fields the
platform sends. Both are therefore attacker-controlled in the threat model the
module's docstring accepts, and the accepted risk is only the one it names - a
compromised *index* serving the right package - if the platform cannot redirect
the installer to some *other* package. Naming the package is choosing what
executes: an sdist runs its build hooks, and a wheel declaring a `nomikos`
console script replaces the entry point `_re_exec` runs next.

These are unit tests on purpose. The end-to-end upgrade lives in
`tests/inference/integration/test_cli_self_upgrade.py`, which needs a platform
and a package index and is skipped wherever those are absent; this file needs
neither, so the guard is checked on every run of the unit lane.
"""

from __future__ import annotations

import pytest

from inference.cli import console as ui
from inference.cli import upgrade as upgrade_module
from inference.cli.api import AgentFloor
from inference.cli.version import DISTRIBUTION_NAME


def _floor(*, package: str = DISTRIBUTION_NAME, minimum_version: str = "9.9.9") -> AgentFloor:
    return AgentFloor(
        agent_version="0.1.0",
        minimum_version=minimum_version,
        latest_version=minimum_version,
        package=package,
        refused=True,
        outdated=False,
        message="This agent is below the floor.",
    )


@pytest.fixture
def installer_that_must_not_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Records any installer invocation. A recorded call is a failed test."""
    attempted: list[list[str]] = []

    def _refuse(command: list[str]):  # pragma: no cover - the assertion is the point
        attempted.append(command)
        raise AssertionError(f"an installer was run: {command}")

    monkeypatch.setattr(upgrade_module, "_install", _refuse)
    # `_re_exec` replaces the process. If a test ever reaches it, it would take
    # the test runner with it.
    monkeypatch.setattr(
        upgrade_module,
        "_re_exec",
        lambda _version: pytest.fail("the agent re-executed itself"),
    )
    return attempted


@pytest.mark.parametrize(
    "package",
    [
        "attacker-pkg",
        "nomikos-inferance",  # a typosquat of the real name
        "nomikos_inference",  # underscore, not the distribution name
        "nomikos-inference[extra]",
        "nomikos-inference --index-url http://example.invalid/simple",
        "",
        "../../../etc/passwd",
    ],
)
def test_a_package_that_is_not_this_distribution_is_never_installed(
    package: str,
    installer_that_must_not_run: list[list[str]],
) -> None:
    exit_code = upgrade_module._upgrade_or_stop(
        ui.out(), ui.err(), _floor(package=package), "0.1.0"
    )

    assert exit_code == ui.EXIT_FAILED
    assert installer_that_must_not_run == []


@pytest.mark.parametrize(
    "minimum_version",
    [
        "0 --index-url http://example.invalid/simple",
        "9.9.9 ; python_version < '4'",
        "latest",
        "",
        "   ",
        "9.9.9\n--extra-index-url http://example.invalid",
        "1" * 64,  # longer than the column that records a version
    ],
)
def test_a_floor_that_is_not_a_version_is_never_installed(
    minimum_version: str,
    installer_that_must_not_run: list[list[str]],
) -> None:
    exit_code = upgrade_module._upgrade_or_stop(
        ui.out(), ui.err(), _floor(minimum_version=minimum_version), "0.1.0"
    )

    assert exit_code == ui.EXIT_FAILED
    assert installer_that_must_not_run == []


@pytest.mark.parametrize(
    "minimum_version",
    ["0.4.0", "1.0.0", "0.10.0", "2.0.0rc1", "0.5.0b2", "1.2.3+cpu", "0.4.0-dev1"],
)
def test_the_platforms_own_version_grammar_is_accepted(minimum_version: str) -> None:
    """The guard must not refuse what the platform can legitimately send.

    This is the same grammar `backend/ml/domain/agent_version.py` accepts. A
    guard that rejected a real floor would turn a routine release into an
    outage for every agent below it.
    """
    assert (
        upgrade_module._reject_unexpected_target(ui.err(), _floor(minimum_version=minimum_version))
        is False
    )


def test_the_expected_distribution_at_a_real_version_reaches_the_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is a filter, not a wall - the legitimate upgrade still runs."""
    requirements: list[str] = []

    monkeypatch.setattr(
        upgrade_module,
        "_installer_command",
        lambda requirement: requirements.append(requirement) or ["true"],
    )
    monkeypatch.setattr(
        upgrade_module,
        "_install",
        lambda _command: __import__("subprocess").CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(upgrade_module, "_re_exec", lambda _version: None)

    upgrade_module._upgrade_or_stop(ui.out(), ui.err(), _floor(minimum_version="0.4.0"), "0.1.0")

    assert requirements == [f"{DISTRIBUTION_NAME}>=0.4.0"]
