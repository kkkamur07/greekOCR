"""A credential goes over TLS, and a page image link may not name a scheme.

Two guards on the same surface, for two different reasons.

Every request the CLI makes carries a bearer credential in a header - the
180-day **device token**, or a **service credential** with wider scope. The
client used to accept whatever URL it was handed, so `--api-url http://...`,
`$NOMICOUS_API_URL`, or a stored `platform_url` put that token on the wire in
cleartext, and the researcher who typed it got no indication.

The page image link is the opposite shape of problem: it carries no credential,
but it is chosen *entirely* by the platform and passed to `urllib.request.urlopen`.
`urlopen` is not an HTTP client. Its default opener services `file:` too, so a
claim response naming `file:///etc/passwd` was read off the researcher's disk and
sent onward as the "page image" - which, since the inline `image_bytes` came out
of the claim response, is now the only path by which an agent gets a scan at all.

Loopback HTTP stays legal in both, and deliberately: the integration suite stands
a platform up on `http://127.0.0.1:<port>`, and nothing leaves the machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.cli.api import (
    REQUEST_TIMEOUT_SECONDS,
    InsecurePlatformURL,
    PlatformClient,
    PlatformError,
)


# ---------------------------------------------------------------------------
# The platform URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.nomicous.com",
        "http://example.invalid:8000",
        # Loopback in the *userinfo*, not the host. The token still leaves the
        # machine; only a check that parses rather than substring-matches sees it.
        "http://localhost@evil.example",
        "http://127.0.0.1.evil.example",
        "http://notlocalhost",
        "ftp://api.nomicous.com",
        "file:///etc/passwd",
        "api.nomicous.com",
        "",
    ],
)
def test_a_client_refuses_to_carry_a_credential_over_this_url(base_url: str) -> None:
    with pytest.raises(InsecurePlatformURL):
        PlatformClient(base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.nomicous.com",
        "https://api.nomicous.com/",
        "https://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:54321",
        "http://[::1]:8000",
        "http://LOCALHOST:8000",
    ],
)
def test_https_anywhere_and_http_on_loopback_are_accepted(base_url: str) -> None:
    """The guard must not refuse what a researcher or the suite legitimately uses."""
    assert PlatformClient(base_url).base_url == base_url.rstrip("/")


def test_the_trailing_slash_is_still_stripped() -> None:
    """The one thing `__init__` did before the check was added still happens."""
    assert PlatformClient("https://api.nomicous.com/").base_url == "https://api.nomicous.com"


# ---------------------------------------------------------------------------
# The page image link
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://localhost/etc/passwd",
        "ftp://example.invalid/page.png",
        "data:image/png;base64,AAAA",
        "/etc/passwd",
        "",
    ],
)
def test_a_page_image_link_that_is_not_https_is_never_opened(url: str) -> None:
    """The platform chooses this string; it must not be able to choose a scheme.

    Before the guard, `urlopen` serviced every one of these. `file:` is the one
    that matters - it read the researcher's disk and returned the bytes as a page
    image - and the fix is a scheme allowlist rather than a `file:` denylist,
    because the next opener Python registers should not silently become reachable.
    """
    client = PlatformClient("https://api.nomicous.com")

    with pytest.raises(PlatformError, match="page image"):
        client.fetch_page_image(url)


def test_the_file_scheme_cannot_read_a_real_file_off_this_machine(tmp_path: Path) -> None:
    """The concrete exploit, pinned: a readable file that is not a page image."""
    secret = tmp_path / "secret.txt"
    secret.write_text("device-token-would-be-here", encoding="utf-8")
    client = PlatformClient("https://api.nomicous.com")

    with pytest.raises(PlatformError):
        client.fetch_page_image(secret.as_uri())

    assert secret.read_text(encoding="utf-8") == "device-token-would-be-here"


def test_a_loopback_platform_may_not_hand_out_a_file_url() -> None:
    """The loopback exemption is about *http*, not about opening local paths."""
    client = PlatformClient("http://127.0.0.1:8000")

    with pytest.raises(PlatformError):
        client.fetch_page_image("file:///etc/passwd")


def test_an_https_platform_may_not_hand_out_a_loopback_http_image() -> None:
    """The exemption is a property of this client, not of the URL it is given.

    Otherwise a hosted platform could point a researcher's agent at a service
    listening on their own machine, which is the SSRF this check exists for.
    """
    client = PlatformClient("https://api.nomicous.com")

    with pytest.raises(PlatformError):
        client.fetch_page_image("http://127.0.0.1:8000/page.png")


# ---------------------------------------------------------------------------
# The long poll
# ---------------------------------------------------------------------------
def test_the_claim_deadline_leaves_room_for_the_wait_it_asked_for() -> None:
    """`--wait-seconds` above ~28 used to time out every claim it made.

    One `REQUEST_TIMEOUT_SECONDS` covered every request including the claim long
    poll, so an agent that asked the platform to hold the connection for its own
    ceiling - settable to 120 - hung up first, every time. The flag's help says
    "Clamped by the platform"; the client was the thing clamping it.
    """
    seen: list[float | None] = []
    client = PlatformClient("https://api.nomicous.com")

    def _capture(method, path, *, body=None, headers=None, timeout=None):
        seen.append(timeout)
        return 200, {"page": None, "poll_after_seconds": 1.0}

    client._request = _capture  # type: ignore[assignment]
    client.claim_page(credential={}, wait_seconds=90)

    assert seen == [90 + REQUEST_TIMEOUT_SECONDS]


def test_an_ordinary_request_keeps_the_ordinary_deadline() -> None:
    """The per-call override is for the long poll alone."""
    seen: list[float | None] = []
    client = PlatformClient("https://api.nomicous.com")

    def _capture(method, path, *, body=None, headers=None, timeout=None):
        seen.append(timeout)
        return 401, None

    client._request = _capture  # type: ignore[assignment]
    client.read_self(device_token="token")

    assert seen == [None]
