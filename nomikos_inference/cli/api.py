"""HTTP client for the platform's device protocols.

Pairing is RFC 8628's device authorization grant with the typable `user_code`
removed (`backend/ml/api/device_pairing.py`); the **claim** loop is the one
endpoint ADR 0003 costs, plus the platform's existing job callback
(`backend/jobs/api/device_claim.py`, `backend/jobs/api/internal_inference.py`).
This is a client for wire contracts that are already settled, not a design
surface, and the shapes below are the platform's DTOs read back.

`urllib.request` rather than an HTTP library, because `[project].dependencies`
is the closure that reaches a researcher's laptop, and neither protocol needs
streaming, connection reuse, or a negotiated auth scheme. One page per claim
(ADR 0002) keeps that true: no long-lived transfer to manage, just a request
per page.
"""

from __future__ import annotations

import json
import os
import platform as platform_module
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from nomikos_inference.cli.version import installed_version

DEFAULT_PLATFORM_URL = "https://api.nomikos.app"
PLATFORM_URL_ENV = "NOMIKOS_API_URL"

DEVICE_TOKEN_HEADER = "X-Nomikos-Device-Token"  # noqa: S105 - a header name, not a token
AGENT_VERSION_HEADER = "X-Nomikos-Agent-Version"
"""Which build of the agent is calling. Checked by the **version floor** on the
**claim** path and on `GET /device/v1/agent/version`
(`backend/ml/api/agent_version.py`); defined here so the run loop and the
launch check report the same version."""

SERVICE_TOKEN_HEADER = "X-Nomikos-Service-Token"  # noqa: S105 - a header name, not a token
"""A hosted worker's **service credential**, in its own header because it
resolves to a different scope than the device token (ADR 0005, decision 1): a
device token claims `local` work on one account, this claims `cloud` work for
the platform. The two must never be interchangeable by accident."""

WORKER_NAME_HEADER = "X-Nomikos-Worker-Name"
"""Which hosted worker is calling. Names its device row; not a secret."""

CLAIM_PATH = "/device/v1/jobs/claim"
CALLBACK_PATH = "/internal/inference/job-complete"

AGENT_VERSION_REFUSED_STATUS = 426
AGENT_VERSION_UNSUPPORTED = "AGENT_VERSION_UNSUPPORTED"
"""The one error code the CLI matches on, in the run loop and the launch check
alike. Stable by contract (the platform's own module says changing it breaks
every agent), and the only refusal that arrives machine-readable, since the
error envelope otherwise replaces `HTTPException.detail` with a fixed public
string."""

IMAGE_TIMEOUT_SECONDS = 120.0
"""Generous because this fetches a scan, not a JSON round trip, but still
bounded: the **signed page image link** dies in about a minute anyway, so a
fetch that hasn't finished well past that isn't going to."""

MAX_PAGE_IMAGE_BYTES_ENV = "NOMIKOS_MAX_PAGE_IMAGE_BYTES"


def max_page_image_bytes() -> int:
    """Ceiling on a downloaded page image, loose and overridable.

    A signed link names exactly one object and expires in about a minute, so
    the practical bound is the platform's own upload cap, not this number. It
    exists only to stop a slow-drip or hostile response from exhausting memory,
    so the default is generous; raise it via ``NOMIKOS_MAX_PAGE_IMAGE_BYTES``
    if needed.
    """
    raw = os.environ.get(MAX_PAGE_IMAGE_BYTES_ENV)
    if raw is None:
        return 1024 * 1024 * 1024
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{MAX_PAGE_IMAGE_BYTES_ENV} must be an integer") from exc
    return max(1, parsed)


# The platform caps these on the way in (`PairingStartRequest`). Truncating
# here keeps a long hostname from turning into a 422 the researcher can't act on.
DEVICE_NAME_LIMIT = 120
PLATFORM_LIMIT = 32
VERSION_LIMIT = 32

REQUEST_TIMEOUT_SECONDS = 30.0

# Pairing states, named as the platform names them (`backend/ml/domain/devices.py`).
# All arrive inside a 200 body: the platform's error envelope replaces
# `HTTPException.detail` with a fixed public string, so a machine-readable
# state can't survive a non-2xx response.
STATUS_AUTHORIZATION_PENDING = "authorization_pending"
STATUS_SLOW_DOWN = "slow_down"
STATUS_ACCESS_DENIED = "access_denied"
STATUS_EXPIRED = "expired"
STATUS_APPROVED = "approved"


LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
"""The only hosts this CLI will talk to without TLS.

Every request on this surface carries a bearer credential (the 180-day
**device token**, or a **service credential** with wider scope) in a header.
Over cleartext HTTP those are readable by anything on the path, so the scheme
is checked once, where the client is built, rather than per call. Loopback is
the exception because nothing leaves the machine: it's what lets the
integration suite stand up a platform on `http://127.0.0.1:<port>`.
"""


class PlatformError(RuntimeError):
    """The platform could not be reached, or answered something unusable."""


class InsecurePlatformURL(PlatformError):
    """A platform URL this CLI will not send a credential to."""


class CredentialRefused(PlatformError):
    """The platform will not accept this agent's credential (401).

    Its own exception because it's the one **claim** failure that backing off
    can't fix: a revoked, expired, or unknown credential answers 401 forever.
    A loop that retried it would hammer the platform until someone noticed;
    stopping and saying `nomikos pair` is the only useful response.
    """


class PageLeaseLost(PlatformError):
    """The platform says this agent is no longer holding the page it is reporting.

    Its own exception, mirroring the reason above: a 403 on the callback is
    terminal for *this* agent. The **lease** expired, the page went back to the
    queue, and someone else may already own it, so retrying the report is at
    best noise and at worst a race with the agent that now holds it.
    """


class AgentVersionRefused(PlatformError):
    """This build is below the **version floor** and may not claim (issue 055).

    Its own exception rather than a status code the loop inspects. `retryable`
    is always false on the wire: no amount of backing off turns this into
    work, so a claim loop that treated it as transient would spin forever
    against a platform that already said what to do.
    """

    def __init__(
        self,
        *,
        message: str,
        agent_version: str | None,
        minimum_version: str,
        latest_version: str,
        upgrade_command: str,
    ) -> None:
        super().__init__(message)
        self.agent_version = agent_version
        self.minimum_version = minimum_version
        self.latest_version = latest_version
        self.upgrade_command = upgrade_command


@dataclass(frozen=True)
class StartedPairing:
    """What `POST /device/v1/pairings` hands an unpaired machine."""

    pairing_id: str
    device_code: str
    verification_url: str
    confirmation_code: str
    """Not a secret and not a `user_code`: no endpoint accepts it. It is a keyed
    derivation of the pairing id, shown here so the researcher can compare it
    with the consent screen (ADR 0001, decision 13)."""
    expires_in: int
    interval_seconds: int


@dataclass(frozen=True)
class PairingPoll:
    """One result from the token collection loop."""

    status: str
    interval_seconds: int
    device_id: str | None = None
    device_token: str | None = None
    token_expires_at: datetime | None = None
    account_email: str | None = None


@dataclass(frozen=True)
class AgentFloor:
    """What the platform makes of this agent's version, asked at launch.

    One shape for both answers: a refusal and a notice carry the same four
    facts (what was presented, the floor, the latest, the package to install)
    and differ only in whether work would have been handed over. Collapsing
    them into one record with two booleans keeps the caller from having to
    catch an exception to learn something the platform already stated.
    """

    agent_version: str
    minimum_version: str
    latest_version: str
    package: str
    refused: bool
    """Below the floor (or unstatable): the platform will not hand this agent a
    **claim** until it upgrades."""
    outdated: bool
    """At or above the floor, behind the latest. Served, and told - a notice, not
    a refusal."""
    message: str = ""
    """The platform's own sentence about it. Printed rather than reworded, so a
    researcher and the server logs say the same thing."""
    upgrade_command: str = ""
    """A hint for a human to read. Never executed: the platform names a package,
    and handing a remote process a server-supplied string to run would be a worse
    bargain than the one ADR 0002 already accepts."""


@dataclass(frozen=True)
class DeviceIdentity:
    """What `GET /device/v1/self` tells a paired machine about itself."""

    device_id: str
    name: str
    account_email: str
    token_expires_at: datetime | None


@dataclass(frozen=True)
class AgentNotice:
    """What the platform makes of this build, delivered *with* the work.

    Arrives on every claim response, page or no page, so an idle agent still
    learns it is behind. **Outdated** is not the same state as being below the
    **version floor**: this one is served normally and merely noted.
    """

    agent_version: str
    latest_version: str
    outdated: bool
    upgrade_command: str


@dataclass(frozen=True)
class ClaimedPage:
    """One page of work, and everything needed to run and report it.

    The run fields are a flattened `JobSubmitRequest`, the same contract the
    inference runtime already takes, which is what keeps a laptop and a hosted
    worker literally the same program (ADR 0003).
    """

    product_job_id: str
    inference_job_id: str
    lease_expires_at: datetime | None
    """When the platform stops considering this agent the owner of the page.
    Read before reporting, to name the cause when the callback is refused."""
    task: str
    registry_model_id: str
    registry_tag: str
    params: dict
    page_image_url: str
    page_image_expires_at: datetime | None
    """Two fields rather than one, so the agent can tell whether its link is
    still worth using without parsing a URL."""

    def image_link_expired(self, *, now: datetime | None = None) -> bool:
        """Whether the **signed page image link** is already dead.

        Checked before the fetch rather than after the 403, so a page whose
        link died (a lid closed mid-claim, a machine that swapped) fails with
        the reason instead of a status the route deliberately makes ambiguous.
        """
        if self.page_image_expires_at is None:
            return False
        return self.page_image_expires_at <= (now or datetime.now(UTC))

    def lease_expired(self, *, now: datetime | None = None) -> bool:
        """Whether this agent's **lease** has run out on the local clock.

        Only used to *explain*, never to skip the callback: the platform is the
        authority on who holds a page, and a laptop with a skewed clock must not
        talk itself out of reporting work it actually finished.
        """
        if self.lease_expires_at is None:
            return False
        return self.lease_expires_at <= (now or datetime.now(UTC))


@dataclass(frozen=True)
class Claim:
    """One answer from the claim endpoint. An empty queue is one of these, not
    an error: it arrives as a 200 with no page, because a healthy platform is
    idle most of the time."""

    page: ClaimedPage | None
    poll_after_seconds: float
    agent: AgentNotice | None


def default_platform_url(environ: dict[str, str] | None = None) -> str:
    source = environ if environ is not None else os.environ
    return (source.get(PLATFORM_URL_ENV) or DEFAULT_PLATFORM_URL).rstrip("/")


def _is_loopback(host: str | None) -> bool:
    """Whether a hostname names this machine. `urlsplit` has already
    lowercased it and stripped the brackets off an IPv6 literal."""
    return bool(host) and host in LOOPBACK_HOSTS


def is_secure_platform_url(base_url: str) -> bool:
    """Whether a bearer credential may be sent to `base_url`."""
    parts = urlsplit(base_url)
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and _is_loopback(parts.hostname)


def require_secure_platform_url(base_url: str) -> str:
    """The trimmed URL, or `InsecurePlatformURL` if it would leak a credential.

    Enforced once, where a client is built, rather than at each call site that
    supplies a URL: `--api-url`, `$NOMIKOS_API_URL`, and the stored credential
    are three doors into the same room, and a check on one of them is a check
    on none of them.
    """
    trimmed = base_url.rstrip("/")
    if is_secure_platform_url(trimmed):
        return trimmed
    scheme = urlsplit(trimmed).scheme or "no"
    raise InsecurePlatformURL(
        f"Refusing to talk to {trimmed or 'an empty URL'} over {scheme} scheme. "
        "Every request carries this machine's device token, so the platform URL "
        "must be https (or http on localhost for a development platform)."
    )


def this_machine_name() -> str:
    """What the consent screen will call this computer.

    Self-reported and rendered as inert text by the `/pair` page: a convincing
    name is not evidence, and the **confirmation code** is what a researcher
    actually checks.
    """
    name = socket.gethostname().strip() or "unnamed machine"
    return name[:DEVICE_NAME_LIMIT]


def this_machine_platform() -> str:
    machine = platform_module.machine() or "unknown"
    return f"{platform_module.system().lower()}-{machine}"[:PLATFORM_LIMIT]


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _read_bounded(response: Any, limit: int) -> bytes:
    """Read a response body, refusing to hold more than ``limit`` bytes.

    Streamed rather than ``response.read()`` so a response whose size was not
    announced (or was lied about) cannot force a single unbounded allocation.
    """
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise PlatformError("the page image exceeded the size limit")
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise PlatformError("the page image exceeded the size limit")
        chunks.append(chunk)


class PlatformClient:
    """Every request the CLI makes: pair a machine, ask the floor, then claim,
    fetch, report."""

    def __init__(self, base_url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = require_secure_platform_url(base_url)
        self.plaintext_loopback = urlsplit(self.base_url).scheme == "http"
        """This client is pointed at a platform on this machine. The only state
        in which it will fetch a page image over cleartext HTTP."""
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, Any]:
        """Send one request; return `(status, decoded body)`.

        Non-2xx statuses are returned rather than raised, since on this surface
        they are answers: a 401 from `/device/v1/self` is how a revoked device
        finds out. Only a failure to reach the platform at all is an exception.

        `timeout` overrides the client's default for the one call that needs
        it: a long poll is *expected* to hold the socket open, so the deadline
        that suits a JSON round trip would abort it every time.
        """
        deadline = self.timeout if timeout is None else timeout
        url = f"{self.base_url}{path}"
        payload = None if body is None else json.dumps(body).encode("utf-8")
        # S310 (here and on the `urlopen` below): `url` is `base_url` + a
        # literal path, and `base_url` passed `require_secure_platform_url`
        # in `__init__` - https, or http to loopback for the integration suite.
        request = urllib.request.Request(url, data=payload, method=method)  # noqa: S310
        request.add_header("Accept", "application/json")
        # Recorded on `helper_pairings.user_agent` for support correlation.
        request.add_header("User-Agent", f"nomikos-inference/{installed_version()}")
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            request.add_header(name, value)

        try:
            with urllib.request.urlopen(request, timeout=deadline) as response:  # noqa: S310
                return response.status, _decode(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _decode(exc.read())
        except urllib.error.URLError as exc:
            raise PlatformError(f"Cannot reach {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PlatformError(f"{self.base_url} did not answer within {deadline:g}s") from exc
        except OSError as exc:  # pragma: no cover - socket-level failures
            raise PlatformError(f"Cannot reach {self.base_url}: {exc}") from exc

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------
    def start_pairing(
        self,
        *,
        device_name: str,
        device_platform: str,
        agent_version: str,
        capabilities: dict | None = None,
    ) -> StartedPairing:
        status, body = self._request(
            "POST",
            "/device/v1/pairings",
            body={
                "device_name": device_name,
                "platform": device_platform,
                "helper_version": agent_version[:VERSION_LIMIT],
                "capabilities": capabilities or {},
            },
        )
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving device pairing. "
                "It is disabled by default in production until the consent page ships."
            )
        if status == 429:
            raise PlatformError(
                f"{self.base_url} is refusing new pairing requests right now. Try again shortly."
            )
        if status != 201 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "starting the pairing"))
        try:
            return StartedPairing(
                pairing_id=body["pairing_id"],
                device_code=body["device_code"],
                verification_url=body["verification_url"],
                confirmation_code=body["confirmation_code"],
                expires_in=int(body["expires_in"]),
                interval_seconds=int(body["interval_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(f"{self.base_url} returned an unusable pairing response") from exc

    def collect_token(self, *, pairing_id: str, device_code: str) -> PairingPoll:
        status, body = self._request(
            "POST",
            "/device/v1/pairings/token",
            body={"pairing_id": pairing_id, "device_code": device_code},
        )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "waiting for approval"))
        try:
            return PairingPoll(
                status=str(body.get("status") or ""),
                interval_seconds=int(body.get("interval_seconds") or 0),
                device_id=body.get("device_id"),
                device_token=body.get("device_token"),
                token_expires_at=_parse_datetime(body.get("token_expires_at")),
                account_email=body.get("account_email"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(f"{self.base_url} returned an unusable pairing response") from exc

    def read_self(self, *, device_token: str) -> DeviceIdentity | None:
        """Confirm a stored credential. `None` means the platform refused it.

        The platform answers every rejection (unknown, expired, revoked) with
        the same 401 and the same public message, so this can't say *why*. The
        caller pairs it with the stored expiry, the one thing it knows
        locally, to name the likely cause.
        """
        status, body = self._request(
            "GET", "/device/v1/self", headers={DEVICE_TOKEN_HEADER: device_token}
        )
        if status == 401:
            return None
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving the device layer, so this "
                "machine's credential cannot be checked."
            )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "checking this machine's credential"))
        return DeviceIdentity(
            device_id=str(body.get("device_id") or ""),
            name=str(body.get("name") or ""),
            account_email=str(body.get("account_email") or ""),
            token_expires_at=_parse_datetime(body.get("token_expires_at")),
        )

    # ------------------------------------------------------------------
    # The claim loop
    # ------------------------------------------------------------------
    def claim_page(self, *, credential: dict[str, str], wait_seconds: int) -> Claim:
        """Take at most one page of work, or come back empty.

        The credential decides the **execution target** and the caller can't
        ask for a different one, so there's nothing to send but how long this
        agent is willing to wait. `wait_seconds` is clamped by the platform,
        not here: the ceiling is an operational dial that belongs to whoever
        is running it.

        The socket deadline is the *requested* wait plus the ordinary request
        budget, not the budget alone. A long poll that asked the platform to
        hold the connection for its own ceiling and then hung up first would
        time out every claim it made, and the flag that asked for it would
        read as broken rather than unsupported.
        """
        wait = max(0, int(wait_seconds))
        status, body = self._request(
            "POST",
            CLAIM_PATH,
            body={"wait_seconds": wait},
            headers=credential,
            timeout=wait + self.timeout,
        )
        if status == AGENT_VERSION_REFUSED_STATUS:
            raise _version_refusal(body, self.base_url)
        if status == 401:
            raise CredentialRefused(
                f"{self.base_url} does not accept this machine's credential. "
                "Run `nomikos pair` to authorise it again."
            )
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving the claim endpoint. "
                "The device layer is disabled by default in production."
            )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "claiming a page"))

        try:
            return Claim(
                page=_claimed_page(body.get("page")),
                poll_after_seconds=float(body.get("poll_after_seconds") or 0.0),
                agent=_agent_notice(body.get("agent")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(f"{self.base_url} returned an unusable claim response") from exc

    def fetch_page_image(self, url: str) -> bytes:
        """Download the one page image a claim points at.

        **No credential is attached, deliberately.** The signature in the URL
        is the authorization (ADR 0002); sending the device token here would
        imply the link needed it, and the link reaches exactly one object
        either way. That's also why this bypasses `_request`: that method adds
        headers this request has no business carrying, and the answer is
        bytes rather than JSON.

        The URL is server-controlled end to end, so its scheme is checked
        before it reaches the opener. `urlopen` is not an HTTP client: the
        default opener also services `file:`, which would turn "here is your
        page image" into a read of any path the researcher's account can open.
        """
        self._require_fetchable(url)
        # S310 (here and on the `urlopen` below): `_require_fetchable` above
        # rejects every scheme but https, which is what stops a server-chosen
        # `file:` URL from reading this machine. See its docstring.
        request = urllib.request.Request(url, method="GET")  # noqa: S310
        request.add_header("User-Agent", f"nomikos-inference/{installed_version()}")
        try:
            with urllib.request.urlopen(request, timeout=IMAGE_TIMEOUT_SECONDS) as response:  # noqa: S310
                return _read_bounded(response, max_page_image_bytes())
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                # Forged, malformed, and expired are one status on purpose, so
                # the route is not an oracle for which object keys exist. Only
                # one of the three can plausibly happen to an honest agent.
                raise PlatformError(
                    "The link to this page's image was refused. It expires about a "
                    "minute after the claim, so it was most likely used too late."
                ) from exc
            raise PlatformError(f"The page image could not be fetched ({exc.code})") from exc
        except urllib.error.URLError as exc:
            raise PlatformError(f"The page image could not be fetched: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PlatformError("The page image did not arrive in time") from exc
        except OSError as exc:  # pragma: no cover - socket-level failures
            raise PlatformError(f"The page image could not be fetched: {exc}") from exc

    def _require_fetchable(self, url: str) -> None:
        """Refuse a page image link that is not an HTTPS URL.

        Plain HTTP is allowed only when this client is itself pointed at a
        loopback platform (the integration suite, and nothing else). The page
        image carries no credential, so this isn't about eavesdropping: it's
        that a URL the platform chose must not be able to name a scheme that
        reads this machine instead of the network.
        """
        parts = urlsplit(url)
        if parts.scheme == "https":
            return
        if parts.scheme == "http" and self.plaintext_loopback and _is_loopback(parts.hostname):
            return
        raise PlatformError(
            f"Refusing to fetch a page image over {parts.scheme or 'a URL with no'} scheme. "
            "A page image link must be https."
        )

    def report_page(
        self,
        *,
        credential: dict[str, str],
        page: ClaimedPage,
        output: dict | None = None,
        error: str | None = None,
    ) -> None:
        """End the page, one way or the other.

        Not a new endpoint: this is the platform's existing
        `JobCallbackRequest` (ADR 0003), authorised by the same credential the
        page was claimed with and narrowed to the page this agent is actually
        holding. A researcher's laptop has no webhook secret and must not be
        given one.
        """
        payload: dict[str, Any] = {
            "inference_job_id": page.inference_job_id,
            "product_job_id": page.product_job_id,
            "task": page.task,
            "status": "done" if error is None else "failed",
        }
        if error is None:
            payload["output"] = output
        else:
            payload["error"] = error

        status, body = self._request("POST", CALLBACK_PATH, body=payload, headers=credential)
        if status == 204:
            return
        if status == 403:
            expired = (
                " Its **lease** had already run out on this machine's clock."
                if (page.lease_expired())
                else ""
            )
            raise PageLeaseLost(
                "The platform says this machine is not holding that page. Its "
                "**lease** most likely expired and the page went back to the "
                f"queue.{expired}"
            )
        raise PlatformError(_unexpected(status, body, "reporting a page"))

    def read_agent_floor(self, *, agent_version: str) -> AgentFloor:
        """Ask what this version is allowed to do, without asking for work.

        Runs the same comparison the **claim** path runs, on an endpoint that
        touches no queue. That separation is the point: an agent learns it's
        below the floor at launch, when nothing is in flight and replacing its
        own code is safe, rather than while holding a page it's already been
        handed.

        No credential is sent or needed: the platform resolves the version
        before it looks at one. So this also answers for a machine that has
        never paired, which is exactly the machine most likely to be stale.
        """
        status, body = self._request(
            "GET", "/device/v1/agent/version", headers={AGENT_VERSION_HEADER: agent_version}
        )
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving the device layer, so it cannot "
                "say which agent version it requires."
            )
        if status == AGENT_VERSION_REFUSED_STATUS:
            return _refusal_floor(self.base_url, agent_version, body)
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "asking for the version floor"))
        try:
            return AgentFloor(
                agent_version=str(body["agent_version"]),
                minimum_version=str(body["minimum_version"]),
                latest_version=str(body["latest_version"]),
                package=str(body["package"]),
                refused=False,
                outdated=bool(body["outdated"]),
                message="",
                upgrade_command=str(body.get("upgrade_command") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(
                f"{self.base_url} returned an unusable version-floor response"
            ) from exc


def _agent_notice(raw: object) -> AgentNotice | None:
    if not isinstance(raw, dict):
        return None
    return AgentNotice(
        agent_version=str(raw.get("agent_version") or ""),
        latest_version=str(raw.get("latest_version") or ""),
        outdated=bool(raw.get("outdated")),
        upgrade_command=str(raw.get("upgrade_command") or ""),
    )


def _claimed_page(raw: object) -> ClaimedPage | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("page must be an object")
    request = raw["request"]
    if not isinstance(request, dict):
        raise TypeError("page.request must be an object")
    params = request.get("params")
    return ClaimedPage(
        product_job_id=str(raw["product_job_id"]),
        inference_job_id=str(raw["inference_job_id"]),
        lease_expires_at=_parse_datetime(raw.get("lease_expires_at")),
        task=str(request["task"]),
        registry_model_id=str(request["registry_model_id"]),
        registry_tag=str(request.get("registry_tag") or "stable"),
        params=params if isinstance(params, dict) else {},
        page_image_url=str(raw["page_image_url"]),
        page_image_expires_at=_parse_datetime(raw.get("page_image_expires_at")),
    )


def _version_refusal(body: Any, base_url: str) -> AgentVersionRefused:
    """Read a 426 back into an exception the run loop can print.

    The refusal is the one failure on this platform that keeps its detail, so
    every field below really is there, but a 426 from something that isn't the
    platform is still possible, so reading it must not raise a `KeyError` on
    top of the refusal.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        error = {}
    return AgentVersionRefused(
        message=str(error.get("message") or f"{base_url} refused this version of the agent"),
        agent_version=error.get("agent_version"),
        minimum_version=str(error.get("minimum_version") or "unknown"),
        latest_version=str(error.get("latest_version") or "unknown"),
        upgrade_command=str(error.get("upgrade_command") or "uv tool upgrade nomikos-inference"),
    )


def _refusal_floor(base_url: str, agent_version: str, body: Any) -> AgentFloor:
    """Read a 426 body into the same record a 200 produces.

    A 426 that doesn't carry the contract is a bug on the platform, not a
    stale agent, and it must not be reported as one: an agent told to upgrade
    with no version to upgrade to would fail loudly for the wrong reason.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != AGENT_VERSION_UNSUPPORTED:
        raise PlatformError(
            f"{base_url} refused this agent's version with {AGENT_VERSION_REFUSED_STATUS} "
            "but did not say what it requires"
        )
    try:
        return AgentFloor(
            agent_version=str(error.get("agent_version") or agent_version),
            minimum_version=str(error["minimum_version"]),
            latest_version=str(error["latest_version"]),
            package=str(error["package"]),
            refused=True,
            outdated=False,
            message=str(error.get("message") or ""),
            upgrade_command=str(error.get("upgrade_command") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlatformError(
            f"{base_url} refused this agent's version but did not say what it requires"
        ) from exc


def _decode(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _unexpected(status: int, body: Any, doing: str) -> str:
    message = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
    detail = f": {message}" if message else ""
    return f"The platform answered {status} while {doing}{detail}"
