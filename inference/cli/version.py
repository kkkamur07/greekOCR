"""`nomikos version` - what this build is, and what it will tell the platform.

The number printed here is not a decoration. Issue 055 put a **version floor**
on the **claim** path: every claim states its version in
`X-Nomikos-Agent-Version`, and an agent below the floor is refused with `426`
before it is authenticated - so it also stops reporting **capacity**. That makes
"which version am I running" the first question worth asking when a machine
stops taking work, and this subcommand is the answer to it.

The floor itself is deliberately not shown, but not because it cannot be: since
`read_agent_floor`, `GET /device/v1/agent/version` answers it without a
credential and without touching a queue, and `nomikos upgrade` - which this same
package runs before every `nomikos run` - asks it there. This subcommand stays
offline anyway. It is the thing a researcher runs when the network is the
suspect, and a version report that needs the platform to answer cannot report
the version of a machine that cannot reach it.
"""

from __future__ import annotations

import argparse
import platform as platform_module
from importlib.metadata import PackageNotFoundError, version

from inference.cli import console as ui

DISTRIBUTION_NAME = "nomikos-inference"
SOURCE_CHECKOUT_VERSION = "0+unknown"


def installed_version() -> str:
    """The version of the installed distribution.

    Read from installed metadata rather than a hardcoded string, so the number
    the CLI reports is the one the resolver actually put on disk. A source
    checkout has no metadata; it says so instead of inventing a version.
    """
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return SOURCE_CHECKOUT_VERSION


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """No options. A version report that takes flags is a version report that
    can be asked the wrong question."""


def run(args: argparse.Namespace) -> int:
    # Imported here to break a cycle, not to defer a cost: `api` imports
    # `installed_version` from this module at *its* module scope, so a top-level
    # import of `api` here - or of `credentials`, which imports `api` - fails on
    # a partially initialised module and takes every subcommand with it. There is
    # nothing to defer either way; `main` has already imported both through the
    # other three commands by the time this runs.
    from inference.cli.api import AGENT_VERSION_HEADER
    from inference.cli.credentials import CredentialError, credential_path, load_credential

    console = ui.out()
    agent_version = installed_version()

    console.print(f"[bold]{DISTRIBUTION_NAME}[/bold] {agent_version}")
    console.print()

    table = ui.detail_table()
    table.add_row("Package", ui.value(DISTRIBUTION_NAME))
    table.add_row("Version", ui.value(agent_version))
    table.add_row("Sent as", ui.value(f"{AGENT_VERSION_HEADER}: {agent_version}"))
    table.add_row("Python", ui.value(platform_module.python_version()))
    table.add_row(
        "Platform", ui.value(f"{platform_module.system().lower()}-{platform_module.machine()}")
    )
    table.add_row("Credential", ui.value(credential_path()))

    try:
        credential = load_credential()
    except CredentialError as exc:
        table.add_row("Paired", ui.value(f"unreadable credential - {exc}"))
    else:
        if credential is None:
            table.add_row("Paired", ui.value("no - run `nomikos pair`"))
        else:
            table.add_row(
                "Paired", ui.value(f"{credential.account_email} at {credential.platform_url}")
            )

    console.print(table)

    if agent_version == SOURCE_CHECKOUT_VERSION:
        # A source checkout cannot claim: the floor refuses a version it cannot
        # parse on the same terms as one that is too old.
        console.print()
        console.print(
            "[yellow]This is a source checkout, not an installed distribution.[/yellow] "
            f"The platform refuses a claim from {SOURCE_CHECKOUT_VERSION!r}; "
            f"install {DISTRIBUTION_NAME} to run work.",
        )

    return ui.EXIT_OK
