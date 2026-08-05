"""Argument parsing and dispatch for the `nomicous` console entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from inference.cli import console as ui
from inference.cli import pair as pair_command
from inference.cli import run as run_command
from inference.cli import version as version_command
from inference.cli.version import DISTRIBUTION_NAME, SOURCE_CHECKOUT_VERSION, installed_version

PROGRAM_NAME = "nomicous"

_DESCRIPTION = "Run nomicous manuscript segmentation and transcription on this machine."

_COMMANDS = {
    "pair": (
        "authorise this machine against your nomicous account",
        pair_command.add_arguments,
        pair_command.run,
    ),
    "run": (
        "claim pages from the platform and run them here",
        run_command.add_arguments,
        run_command.run,
    ),
    "version": (
        "report the version this agent presents to the platform",
        version_command.add_arguments,
        version_command.run,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=_DESCRIPTION)
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROGRAM_NAME} {installed_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (summary, add_arguments, _) in _COMMANDS.items():
        subparser = subparsers.add_parser(name, help=summary, description=summary)
        add_arguments(subparser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point. Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return ui.EXIT_OK

    handler = _COMMANDS[args.command][2]
    try:
        return handler(args)
    except KeyboardInterrupt:
        # A backstop, not the handler that matters. `pair` writes its credential
        # once at the end through a rename, so nothing partial survives; `run`
        # catches its own interrupt, because it may be holding a page and has to
        # report it before it stops.
        ui.err().print("\nInterrupted. Nothing was changed on this machine.")
        return ui.EXIT_INTERRUPTED


__all__ = [
    "DISTRIBUTION_NAME",
    "PROGRAM_NAME",
    "SOURCE_CHECKOUT_VERSION",
    "build_parser",
    "installed_version",
    "main",
]
