"""The `nomicous` console entry point.

ADR 0002 replaced the loopback helper with a CLI installed from PyPI. This
package is the entry point that ships with `nomicous-inference`; the
subcommands that make it useful - `pair`, `version`, `run` - land in #56, #57
and #58. What exists here is deliberately the whole shape and none of the
behaviour, so the boundary the wheel establishes is testable before there is
anything to test through it.
"""

from inference.cli.main import PROGRAM_NAME, installed_version, main

__all__ = [
    "PROGRAM_NAME",
    "installed_version",
    "main",
]
