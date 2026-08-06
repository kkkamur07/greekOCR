"""The `nomicous` console entry point.

ADR 0002 replaced the loopback helper with a CLI installed from PyPI. This
package is that CLI: `pair` authorises a machine against a researcher's account
and stores the **device token** it gets back, `run` is the **claim** loop that
credential exists for, `version` reports the string this agent presents to the
**version floor**, and `upgrade` is the launch check that brings this agent up
to that floor before it claims anything.

Nothing here imports the model runtime at module scope - `run` reaches for it
one line deep, inside the call that needs it. `nomicous version` on a laptop
with no weights and nothing to run should not pay for Torch to answer.
"""

from inference.cli.main import PROGRAM_NAME, installed_version, main

__all__ = [
    "PROGRAM_NAME",
    "installed_version",
    "main",
]
