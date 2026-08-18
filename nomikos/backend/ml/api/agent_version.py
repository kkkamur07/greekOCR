"""The version floor, on the wire: one header in, one refusal or one notice out.

This is the contract the CLI (issue 058) is built against, so the shapes below
matter more than the plumbing behind them.

What the agent sends
--------------------
``X-Nomikos-Agent-Version: 0.4.1`` - a header rather than a body field, for the
same reason the credentials are headers: it identifies the caller, it is needed
before the body is looked at, and ``POST /device/v1/jobs/claim`` accepts an empty
body today. Sending nothing is refused (see :mod:`backend.ml.domain.agent_version`).

What it gets back when it is too old
------------------------------------
``426 Upgrade Required``, in the platform's standard error envelope with a code
no other failure uses::

    {"error": {
        "code": "AGENT_VERSION_UNSUPPORTED",
        "message": "This agent is 0.3.0, and the platform no longer accepts ...",
        "reason": "below_floor",           # or "missing" | "malformed"
        "agent_version": "0.3.0",          # null when nothing was sent
        "minimum_version": "0.4.0",
        "latest_version": "0.6.2",
        "package": "nomikos-inference",
        "upgrade_command": "uv tool upgrade nomikos-inference",
        "retryable": false
    }}

Three things make it actionable rather than merely a failure. ``retryable`` is
``false``, so a claim loop knows this is not a blip to back off from; the status
is 426 and not 401/403/404, so it cannot be confused with a bad credential or a
disabled device layer; and ``minimum_version`` plus ``package`` say exactly what
would fix it. An agent that upgrades and re-execs (ADR 0002) has everything it
needs from this one response.

The platform names a *package*, not a command to run. ``upgrade_command`` is
there to be printed to a human; the machine-usable field is ``package``. Handing
a remote process a server-controlled string to execute would be a worse bargain
than the one ADR 0002 already accepts.

What it gets back when it is merely behind
------------------------------------------
A normal ``200`` claim response, carrying an :class:`AgentVersionNotice` on
``agent``. Every claim response has one, page or no page, so an idle agent still
learns it is behind.

Asking without asking for work
------------------------------
``GET /device/v1/agent/version`` answers the same verdict - the same 426 or the
same notice - and takes nothing from the queue. ADR 0002 gives the agent a
*launch moment*, before any page is in flight, as the one point where it may
swap its own code; issue 058 needs the verdict there, and the claim endpoint
cannot supply it without also handing over a page the agent would then be
holding while it upgraded. Two states, one comparison, two ways in.

It is deliberately unauthenticated, because the check it runs already is: the
version dependency is resolved before any credential is looked at, so an
unpaired laptop that has just been installed can find out it is too old before
it is anything else. It discloses the floor, the latest release, and the package
name - all three of which a 426 already hands to any unauthenticated caller, and
none of which is a secret. It opens no database session.

Where the check sits
--------------------
Before authentication, as a route dependency. Two consequences, both wanted: a
refused agent costs no database work, and it does not get its ``last_seen_at``
touched - so it stops reporting **capacity**, and submission announces "no host
available" instead of creating pages nobody may claim. An agent below the floor
disappears from the platform rather than accumulating work it cannot take.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from backend.core.settings.device import DeviceSettings, get_device_settings
from backend.ml.api.device_dependencies import require_device_pairing_enabled
from backend.ml.domain.agent_version import AgentVersionVerdict, evaluate_agent_version

AGENT_VERSION_HEADER = "X-Nomikos-Agent-Version"
"""Which build of the agent is calling. Not a secret and not a credential - it
decides whether the caller may claim at all, never what it may claim."""

AGENT_VERSION_UNSUPPORTED = "AGENT_VERSION_UNSUPPORTED"
"""The one string a CLI matches on. Stable; changing it breaks every agent."""

AGENT_VERSION_REFUSED_STATUS = 426
"""``426 Upgrade Required``. Chosen over 403 because it says the one thing the
agent needs to know - the credential was never the problem."""


class AgentVersionNotice(BaseModel):
    """What the platform makes of a served agent's version.

    Present on every claim response. ``outdated`` is the only field a CLI has to
    branch on; the rest are for the line it prints.
    """

    agent_version: str = Field(description="The version this agent presented.")
    minimum_version: str = Field(
        description="Below this, the next claim is refused with 426. Read it every poll."
    )
    latest_version: str = Field(description="Newest published agent.")
    outdated: bool = Field(
        description=(
            "True when the agent is at or above the floor but behind the latest. It is "
            "still being served: this is a notice, not a refusal."
        )
    )
    package: str = Field(description="Distribution to upgrade.")
    upgrade_command: str = Field(description="Human-facing hint; print it, do not exec it.")


class AgentVersionRefusal(BaseModel):
    """Body of a 426 - the ``error`` member of the platform's error envelope."""

    code: str = Field(default=AGENT_VERSION_UNSUPPORTED)
    message: str
    reason: str = Field(description="missing | malformed | below_floor")
    agent_version: str | None = Field(
        default=None, description="Echo of what was sent; null when nothing was."
    )
    minimum_version: str
    latest_version: str
    package: str
    upgrade_command: str
    retryable: bool = Field(
        default=False,
        description="Always false. Retrying the same version cannot succeed; upgrading can.",
    )


class AgentVersionRefusalResponse(BaseModel):
    """The 426 body as it goes out - the standard envelope, with a richer member.

    Same ``{"error": {...}}`` shape as every other failure on the platform, so a
    client that already parses one parses this; the extra fields are additions to
    :class:`~backend.core.schemas.errors.ApiErrorDetail`, not a second envelope.
    """

    error: AgentVersionRefusal


class AgentVersionRefusedError(Exception):
    """Raised by the dependency, rendered by the app's handler as a 426.

    It carries the whole body rather than a code, because the platform's error
    envelope deliberately discards ``HTTPException.detail`` and substitutes a
    fixed public message - a machine-readable state cannot survive that path, and
    the CLI needs one.
    """

    def __init__(self, refusal: AgentVersionRefusal) -> None:
        super().__init__(refusal.message)
        self.refusal = refusal


def upgrade_command_for(package: str) -> str:
    return f"uv tool upgrade {package}"


def _notice(verdict: AgentVersionVerdict, settings: DeviceSettings) -> AgentVersionNotice:
    return AgentVersionNotice(
        agent_version=verdict.presented or "",
        minimum_version=verdict.minimum,
        latest_version=verdict.latest,
        outdated=verdict.outdated,
        package=settings.inference_agent_package,
        upgrade_command=upgrade_command_for(settings.inference_agent_package),
    )


def _refusal(verdict: AgentVersionVerdict, settings: DeviceSettings) -> AgentVersionRefusal:
    return AgentVersionRefusal(
        message=verdict.message(),
        reason=verdict.status.value,
        agent_version=verdict.presented,
        minimum_version=verdict.minimum,
        latest_version=verdict.latest,
        package=settings.inference_agent_package,
        upgrade_command=upgrade_command_for(settings.inference_agent_package),
    )


def require_supported_agent_version(
    x_nomikos_agent_version: Annotated[
        str | None,
        Header(
            alias=AGENT_VERSION_HEADER,
            # Deliberately no ``max_length``: a length constraint here would answer
            # an over-long header with the generic 422 envelope, and every rejected
            # version has to come back as the same 426 contract or the CLI has two
            # failure shapes to parse instead of one. Length is checked where the
            # rest of the grammar is.
            description="Agent version, MAJOR.MINOR.PATCH. Required: an agent that does "
            "not say what it is cannot claim.",
        ),
    ] = None,
) -> AgentVersionNotice:
    """Refuse an agent below the floor; tell one that is merely behind.

    Returns the notice so the route can put it on the response it was going to
    send anyway. A dependency rather than a line in the handler so the refusal
    lands before the endpoint body - and therefore before any session is opened.
    """
    settings = get_device_settings()
    verdict = evaluate_agent_version(
        x_nomikos_agent_version,
        minimum=settings.inference_agent_min_version,
        latest=settings.agent_latest_version(),
    )
    if verdict.refused:
        raise AgentVersionRefusedError(_refusal(verdict, settings))
    return _notice(verdict, settings)


SupportedAgentVersion = Annotated[AgentVersionNotice, Depends(require_supported_agent_version)]
"""Route annotation: refuses a stale agent, and hands the route the notice."""


router = APIRouter(tags=["device-jobs"], dependencies=[Depends(require_device_pairing_enabled)])


@router.get(
    "/device/v1/agent/version",
    response_model=AgentVersionNotice,
    responses={
        AGENT_VERSION_REFUSED_STATUS: {
            "model": AgentVersionRefusalResponse,
            "description": (
                "The agent is below the version floor, or did not say what version it "
                "is. It must upgrade before it claims; retrying the same build cannot "
                "succeed."
            ),
        }
    },
)
def read_agent_version(agent: SupportedAgentVersion) -> AgentVersionNotice:
    """The version verdict on its own, with no page attached.

    The whole endpoint is its dependency. An agent below the floor never reaches
    this body - it gets the same 426 the claim path would have given it, from the
    same comparison - and one that is served gets the same notice the claim
    response would have carried.

    This is what makes ADR 0002's launch check possible. An agent that had to
    claim in order to learn it was stale would be holding a page at the exact
    moment it decided to replace its own code, which is the thing that must never
    happen: the launch moment is the only safe one because nothing is in flight
    during it.
    """
    return agent
