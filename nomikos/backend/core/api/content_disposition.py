"""``Content-Disposition`` for downloads whose names may not be ASCII.

HTTP header values are Latin-1; Starlette raises on anything else. A document
titled in Greek therefore cannot be named with a bare ``filename=``. RFC 6266
solves this with a plain ASCII fallback plus a percent-encoded UTF-8
``filename*``; every current browser reads the latter and saves the real name.
"""

from __future__ import annotations

import re
from urllib.parse import quote

_NON_ASCII = re.compile(r"[^\x20-\x7e]+")
_QUOTE_UNSAFE = re.compile(r'["\\]')


def attachment_disposition(filename: str) -> str:
    """``attachment; filename="..."; filename*=UTF-8''...`` for ``filename``.

    The fallback keeps the extension intact and swaps each run of non-ASCII
    characters for one underscore, so a client that ignores ``filename*`` still
    gets a usable name with the right type. When the name is pure ASCII the two
    forms agree and the ``filename*`` is omitted.
    """
    fallback = _QUOTE_UNSAFE.sub("", _NON_ASCII.sub("_", filename))
    header = f'attachment; filename="{fallback}"'
    if fallback == filename:
        return header
    return f"{header}; filename*=UTF-8''{quote(filename, safe='')}"
