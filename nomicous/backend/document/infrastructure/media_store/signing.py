"""Sign a link to exactly one stored object, good for about one minute.

This is the local media store's half of the **signed page image link** (ADR
0002). Supabase Storage already mints one of these; a filesystem does not, so the
platform signs its own and serves the bytes back on a route where *the signature
is the authorization* - there is no session, no device token, and no ownership
re-derivation behind it.

Two properties do all the work here:

**The signature covers the object key.** Not a bucket, not a prefix, not the
document - the one key, verbatim. Changing a single character of the path
invalidates the signature, so a link to page 1 cannot be walked into a link to
page 2, and a link to ``parts/<uuid>.webp`` cannot be truncated into a listing of
``parts/``.

**The signature covers the expiry.** The expiry travels in the clear because it
has to; moving it forward changes the signed message, so a holder cannot extend
their own link.

Accepted risk, recorded rather than mitigated (ADR 0002): the signature is a
bearer token in a URL, so it leaks through access logs and crash dumps. It is
bounded to one object and one minute, which is the whole mitigation.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from hashlib import sha256
from urllib.parse import quote

from backend.core.settings import get_storage_settings
from backend.document.infrastructure.media_store.keys import validate_image_key

#: Where the platform serves objects it signed itself. Deliberately not under
#: ``/device/`` - nothing about this route is device-scoped, and putting it there
#: would suggest the device credential has something to do with reaching it.
SIGNED_MEDIA_PREFIX = "/media/signed"


def _message(image_key: str, expires: int) -> bytes:
    # ``\n`` is not a legal character in an image key (see ``_SAFE_IMAGE_KEY``),
    # so the two fields cannot be shifted across the separator to forge a
    # different (key, expiry) pair with the same digest.
    return f"{image_key}\n{expires}".encode()


def sign_object_path(image_key: str, *, expires_at: datetime) -> str:
    """Root-relative signed path for exactly *image_key*, dead after *expires_at*.

    Takes an absolute deadline rather than a TTL on purpose: it makes the issuing
    moment an argument, so an expired link can be minted deliberately by a test
    without waiting a minute for one to rot.
    """
    validate_image_key(image_key)
    expires = int(expires_at.timestamp())
    signature = hmac.new(
        get_storage_settings().url_signing_key().encode(),
        _message(image_key, expires),
        sha256,
    ).hexdigest()
    return f"{SIGNED_MEDIA_PREFIX}/{quote(image_key)}?expires={expires}&signature={signature}"


def signature_is_valid(
    image_key: str, *, expires: int, signature: str, now: datetime | None = None
) -> bool:
    """Whether *signature* authorizes reading *image_key* right now.

    Returns ``False`` rather than raising for every way a request can be wrong -
    unparseable key, forged digest, expired deadline - because the caller answers
    all three with the same 403. Telling them apart would turn the route into an
    oracle for which object keys exist.
    """
    try:
        validate_image_key(image_key)
    except ValueError:
        return False
    if expires <= int((now or datetime.now(UTC)).timestamp()):
        return False
    expected = hmac.new(
        get_storage_settings().url_signing_key().encode(),
        _message(image_key, expires),
        sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
