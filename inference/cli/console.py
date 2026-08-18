"""What the CLI prints on, and what it exits with.

Two Consoles, deliberately: everything a researcher is meant to read goes to
stdout, and everything that explains a non-zero exit goes to stderr, so
`nomikos pair > pairing.txt` still shows the failure.

`highlight=False` everywhere. Rich's automatic highlighter colours anything that
looks like a number, a path, or a URL, which on this surface means it recolours
the **confirmation code** and the pairing URL according to what characters they
happen to contain. Those two strings are the ones a researcher compares against
a web page; they get one deliberate style each, not an incidental one.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

EXIT_OK = 0
EXIT_FAILED = 1
"""Pairing was refused, expired, or this machine's credential is no longer
accepted. Anything a researcher has to act on before trying again."""
EXIT_INTERRUPTED = 130
"""Ctrl-C, by the shell convention (128 + SIGINT)."""


def out() -> Console:
    return Console(highlight=False)


def err() -> Console:
    return Console(stderr=True, highlight=False)


def unbroken(console: Console, text: str) -> None:
    """Print something that must survive the terminal being narrower than it.

    Rich wraps at the console width, and a wrapped pairing URL is a URL a
    researcher cannot copy in one go. `soft_wrap` turns off wrapping *and*
    cropping for this one line, so it runs on and the terminal decides.
    """
    console.print(text, soft_wrap=True)


def detail_table() -> Table:
    """A borderless label/value block, the CLI's one layout for reporting facts.

    The value column folds rather than crops: these cells hold credential paths
    and device ids, and a path with an ellipsis in the middle of it is worse
    than no path at all.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="left")
    table.add_column(justify="left", overflow="fold")
    return table


def value(raw: object) -> Text:
    """A cell that is never parsed as Rich markup.

    Account emails and device names arrive from the platform, and the hostname
    arrives from the machine. None of them is trusted to contain `[bold]`.
    """
    return Text(str(raw))
