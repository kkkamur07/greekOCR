"""`python -m inference.cli`, the same entry point the console script calls.

The installed `nomikos` script is the interface a researcher uses. This exists
so a source checkout can run the CLI without installing itself first.
"""

from __future__ import annotations

import sys

from inference.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
