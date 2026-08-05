"""Argument parsing and dispatch for the `nomicous` console entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

PROGRAM_NAME = "nomicous"
DISTRIBUTION_NAME = "nomicous-inference"
SOURCE_CHECKOUT_VERSION = "0+unknown"

_DESCRIPTION = "Run nomicous manuscript segmentation and transcription on this machine."

# Named here rather than left to a bare "no subcommands" message: a researcher
# who installs the package before the run loop exists should be told what the
# command is going to do, not that it does nothing.
_PENDING_SUBCOMMANDS = (
    ("pair", "authorise this machine against your nomicous account"),
    ("run", "claim pages from the platform and run them here"),
    ("version", "report the installed version and the platform's version floor"),
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=_DESCRIPTION)
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROGRAM_NAME} {installed_version()}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point. Returns the process exit status."""
    parser = build_parser()
    parser.parse_args(argv)

    print(f"{PROGRAM_NAME} {installed_version()}")
    print()
    print("No subcommands are available yet. Planned:")
    for name, summary in _PENDING_SUBCOMMANDS:
        print(f"  {name:<8} {summary}")
    return 0
