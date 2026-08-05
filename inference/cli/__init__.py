"""The `nomicous` console entry point.

ADR 0002 replaced the loopback helper with a CLI installed from PyPI. This
package is that CLI: `pair` authorises a machine against a researcher's account
and stores the **device token** it gets back, and `version` reports the string
this agent presents to the **version floor**. `run` - the **claim** loop - lands
in #57 on top of the credential `pair` writes.

Nothing here imports the model runtime. `nomicous version` on a laptop with no
weights and nothing to run should not pay for Torch to answer.
"""

from inference.cli.main import PROGRAM_NAME, installed_version, main

__all__ = [
    "PROGRAM_NAME",
    "installed_version",
    "main",
]
