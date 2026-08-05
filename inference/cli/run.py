"""`nomicous run` - the **claim** loop: take a page, run it here, report it, repeat.

This closes the four-step path ADR 0003 is built around - enqueue, claim, run,
callback - across one database and one HTTP hop, with no inbound connection and
no open port. Everything it talks to already existed except the claim itself:
completion and failure are the platform's `JobCallbackRequest`, and abandonment
is the stale sweep.

Three properties are the whole design.

**One page in flight, always.** A batch is N claims (ADR 0002), so the loop holds
exactly one page between claiming it and reporting it. That is what makes a
**lease** sufficient without a heartbeat, what makes progress free, and what
bounds the damage of a closed lid to one page rather than a document.

**Every page ends, and ends terminally.** A page that fails to run here is
reported *failed with its reason* and the loop moves on, because a researcher
watching a job must never be left waiting on a page that already died. Ctrl-C
reports the page in flight before exiting for the same reason - a considerate
shutdown leaves nothing stuck. What a crash leaves behind is covered by the
lease, which re-queues the page rather than failing it: a closed lid is not a
failed job.

**Local and cloud are the same program.** A hosted worker runs this loop with a
**service credential** and a short poll instead of a **device token** and a long
one. That is the only difference, and it is two lines below rather than a second
code path kept in parity by discipline.

Nothing at module scope imports the model runtime. `inference/cli/main.py`
imports this module to build its parser, so a top-level `run_model` import would
make `nomicous version` on a laptop with no weights pay for Torch to answer.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

from inference.cli import console as ui
from inference.cli.api import (
    AGENT_VERSION_HEADER,
    DEVICE_TOKEN_HEADER,
    SERVICE_TOKEN_HEADER,
    WORKER_NAME_HEADER,
    AgentNotice,
    AgentVersionRefused,
    ClaimedPage,
    PlatformClient,
    PlatformError,
    default_platform_url,
    this_machine_name,
)
from inference.cli.credentials import CredentialError, credential_path, load_credential
from inference.cli.version import installed_version

SERVICE_TOKEN_ENV = "NOMICOUS_SERVICE_TOKEN"
"""A hosted worker's **service credential**, read from the environment and never
from a flag. A token passed on the command line is a token in `ps` output and in
shell history, and this one is not bounded by a single account the way a device
token is."""

WORKER_NAME_ENV = "NOMICOUS_WORKER_NAME"

LAPTOP_WAIT_SECONDS = 25
"""A researcher's page should start within a second of being submitted, so a
laptop long-polls. The platform clamps this to its own ceiling."""

WORKER_WAIT_SECONDS = 0
"""A hosted worker short-polls instead: it is never idle for long, it does not
need the latency, and every second it waits is a request-handler slot held on a
serverless host (ADR 0003)."""

INTERRUPTED_ERROR = "Stopped on the machine running it before the page finished"

MAX_REASON_CHARS = 160
"""The platform prefixes the reason, redacts URLs and paths out of it, and
truncates at 200. Staying well inside that is what keeps a reason readable
instead of ending in a placeholder."""


class RunSetupError(RuntimeError):
    """This machine cannot start claiming, and says why."""


@dataclass(frozen=True)
class AgentIdentity:
    """Which credential this process claims with, and what that credential means.

    The **execution target** is not negotiable and is not sent: the credential
    fixes it (ADR 0005, decision 1). This records it only so the loop can say
    out loud whose work it is about to take.
    """

    credential: dict[str, str]
    account: str
    execution_target: str
    default_wait_seconds: int


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-url",
        metavar="URL",
        help="Platform to claim from. Defaults to $NOMICOUS_API_URL, then the hosted API.",
    )
    parser.add_argument(
        "--exit-when-empty",
        action="store_true",
        help=(
            "Stop once the queue has nothing left, instead of waiting for more. "
            "The scripted half of this command; without it, it keeps running."
        ),
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        metavar="SECONDS",
        help=(
            "How long one claim waits for work before coming back empty. "
            f"Defaults to {LAPTOP_WAIT_SECONDS} for a paired machine and "
            f"{WORKER_WAIT_SECONDS} for a hosted worker. Clamped by the platform."
        ),
    )


def run(args: argparse.Namespace) -> int:
    console = ui.out()
    errors = ui.err()

    base_url = (args.api_url or default_platform_url()).rstrip("/")
    try:
        identity = _resolve_identity(base_url)
    except RunSetupError as exc:
        errors.print(f"[red]{exc}[/red]")
        return ui.EXIT_FAILED

    wait_seconds = (
        identity.default_wait_seconds if args.wait_seconds is None else max(0, args.wait_seconds)
    )
    client = PlatformClient(base_url)
    _announce(console, identity, base_url)

    try:
        return _claim_loop(
            console,
            errors,
            client,
            identity,
            wait_seconds=wait_seconds,
            exit_when_empty=args.exit_when_empty,
        )
    except AgentVersionRefused as exc:
        _report_refusal(errors, exc)
        return ui.EXIT_FAILED
    except PlatformError as exc:
        errors.print(f"[red]{exc}[/red]")
        return ui.EXIT_FAILED


# ---------------------------------------------------------------------------
# Which credential this process claims with
# ---------------------------------------------------------------------------
def _resolve_identity(base_url: str) -> AgentIdentity:
    """A **service credential** if one is in the environment, else this machine's.

    Service-first, matching the order the platform resolves them in
    (`resolve_inference_agent`): a host that has been given a service token is a
    hosted worker, and a stray `device.json` in its home directory must not
    quietly turn it into somebody's laptop.
    """
    version = installed_version()
    service_token = (os.environ.get(SERVICE_TOKEN_ENV) or "").strip()
    if service_token:
        worker_name = (os.environ.get(WORKER_NAME_ENV) or "").strip() or this_machine_name()
        return AgentIdentity(
            credential={
                SERVICE_TOKEN_HEADER: service_token,
                WORKER_NAME_HEADER: worker_name,
                AGENT_VERSION_HEADER: version,
            },
            account=f"the platform, as {worker_name}",
            execution_target="cloud",
            default_wait_seconds=WORKER_WAIT_SECONDS,
        )

    try:
        credential = load_credential()
    except CredentialError as exc:
        raise RunSetupError(
            f"{exc}\nDelete {credential_path()} and run `nomicous pair` again to replace it."
        ) from exc
    if credential is None:
        raise RunSetupError(
            "This machine is not paired, so it has nothing to claim work with.\n"
            "Run `nomicous pair` to authorise it against your account."
        )
    if credential.platform_url != base_url:
        # The credential is only meaningful against the platform that minted it,
        # and presenting it elsewhere is a 401 with no explanation attached.
        raise RunSetupError(
            f"This machine is paired with {credential.platform_url}, not {base_url}.\n"
            "Run `nomicous pair --force` to pair it with this platform instead."
        )

    return AgentIdentity(
        credential={
            DEVICE_TOKEN_HEADER: credential.device_token,
            AGENT_VERSION_HEADER: version,
        },
        account=credential.account_email or "this account",
        execution_target="local",
        default_wait_seconds=LAPTOP_WAIT_SECONDS,
    )


def _announce(console, identity: AgentIdentity, base_url: str) -> None:
    console.print(f"[bold]nomicous run[/bold] {installed_version()}")
    console.print()
    table = ui.detail_table()
    table.add_row("Platform", ui.value(base_url))
    table.add_row("Claiming", ui.value(f"{identity.execution_target} work for {identity.account}"))
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def _claim_loop(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    *,
    wait_seconds: int,
    exit_when_empty: bool,
) -> int:
    done = 0
    failed = 0
    claimed = 0
    told_outdated = False
    waiting_announced = False

    while True:
        try:
            claim = client.claim_page(credential=identity.credential, wait_seconds=wait_seconds)
        except KeyboardInterrupt:
            # Nothing is in flight between claims, which is the entire reason the
            # loop is shaped this way: there is no page here to report.
            console.print()
            _summarise(console, done, failed, "Stopped.")
            return ui.EXIT_INTERRUPTED

        if claim.agent is not None and claim.agent.outdated and not told_outdated:
            told_outdated = True
            _report_outdated(console, claim.agent)

        if claim.page is None:
            if exit_when_empty:
                _summarise(console, done, failed, "Nothing left in the queue.")
                return ui.EXIT_OK
            if not waiting_announced:
                waiting_announced = True
                console.print("[dim]Waiting for work. Press Ctrl-C to stop.[/dim]")
            try:
                time.sleep(max(claim.poll_after_seconds, 0.5))
            except KeyboardInterrupt:
                console.print()
                _summarise(console, done, failed, "Stopped.")
                return ui.EXIT_INTERRUPTED
            continue

        waiting_announced = False
        try:
            outcome = _handle_page(console, errors, client, identity, claim.page, claimed + 1)
        except KeyboardInterrupt:
            # Reachable only in the handful of bytecodes between the claim
            # returning and `_handle_page` taking responsibility for the page -
            # it does not raise this itself. Reporting here is what closes the
            # one window in which a page could be claimed and never ended.
            console.print()
            _report(
                console,
                errors,
                client,
                identity,
                claim.page,
                output=None,
                reason=INTERRUPTED_ERROR,
            )
            _summarise(console, done, failed + 1, "Stopped.")
            return ui.EXIT_INTERRUPTED

        claimed += 1
        if outcome.finished:
            done += 1
        else:
            failed += 1
        if outcome.stopped:
            console.print()
            _summarise(console, done, failed, "Stopped.")
            return ui.EXIT_INTERRUPTED


@dataclass(frozen=True)
class PageOutcome:
    """What became of one page, and whether the loop should keep going."""

    finished: bool
    stopped: bool
    """A Ctrl-C arrived while this page was in flight. The page has already been
    reported by the time this is read - stopping is the only thing left to do."""


def _handle_page(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    page: ClaimedPage,
    index: int,
) -> PageOutcome:
    """Run one page and report it, whatever happens to it.

    This deliberately never raises `KeyboardInterrupt`. Reporting the page is
    the *last* thing that must survive an interrupt, so an exception that skips
    past the callback would be the one bug this whole design exists to prevent -
    a researcher left watching a page nobody is running any more.
    """
    output: dict | None = None
    reason: str | None = None
    interrupted = False
    started = time.monotonic()

    try:
        console.print(
            f"[bold][{index}][/bold] {page.task} [cyan]{page.registry_model_id}[/cyan] "
            f"job {_short(page.product_job_id)}"
        )
        image = client.fetch_page_image(page.page_image_url)
        console.print(f"    fetched {_bytes(len(image))}")
        output = _job_output(page, _execute(page, image))
    except KeyboardInterrupt:
        interrupted = True
        reason = INTERRUPTED_ERROR
    except Exception as exc:  # noqa: BLE001 - one bad page is not a bad loop
        reason = _reason(exc)

    elapsed = time.monotonic() - started
    if reason is None:
        console.print(f"    ran in {elapsed:.1f}s")
    else:
        console.print(f"    [red]failed after {elapsed:.1f}s:[/red] {reason}")

    _report(console, errors, client, identity, page, output=output, reason=reason)
    return PageOutcome(finished=reason is None, stopped=interrupted)


def _report(
    console,
    errors,
    client: PlatformClient,
    identity: AgentIdentity,
    page: ClaimedPage,
    *,
    output: dict | None,
    reason: str | None,
) -> None:
    """Post the terminal callback, absorbing one Ctrl-C while it is in flight.

    The first extra interrupt is swallowed and explained, because a researcher
    pressing Ctrl-C twice in a second means "stop", not "leave that page
    half-reported". The second is honoured: at that point the **lease** is the
    right mechanism, and it re-queues the page rather than failing it.
    """
    for last_attempt in (False, True):
        try:
            client.report_page(
                credential=identity.credential, page=page, output=output, error=reason
            )
        except KeyboardInterrupt:
            if last_attempt:
                errors.print(
                    "    [yellow]Abandoned while reporting.[/yellow] The platform releases "
                    "this page when its lease expires, and another agent may take it."
                )
                return
            console.print(
                "    [dim]still reporting this page - press Ctrl-C again to abandon it[/dim]"
            )
            continue
        except PlatformError as exc:
            errors.print(f"    [red]{exc}[/red]")
            return
        console.print(f"    reported {'done' if reason is None else 'failed'}")
        return


# ---------------------------------------------------------------------------
# Running the page
# ---------------------------------------------------------------------------
def _execute(page: ClaimedPage, image_bytes: bytes):
    """Call the same `run_model` the platform's own worker calls.

    Imported here rather than at module scope: this is the only line in the CLI
    that needs Torch, and `nomicous version` must not pay for it.
    """
    from inference.contracts.common import InferenceTask
    from inference.jobs.runner import run_model

    return run_model(
        task=InferenceTask(page.task),
        registry_model_id=page.registry_model_id,
        registry_tag=page.registry_tag,
        image_bytes=image_bytes,
        params=page.params,
    )


def _job_output(page: ClaimedPage, result) -> dict:
    """Wrap the model's answer in the callback's discriminated union.

    `kind` has to equal the task or the platform rejects the callback, so this
    also catches the one shape mismatch that can happen honestly: a transcribe
    page whose line regions did not survive, which comes back as a single-line
    response where a batch was required.
    """
    data = result.model_dump(mode="json")
    if page.task == "transcribe" and "lines" not in data:
        raise RunSetupError("this page carried no line regions to transcribe")
    return {"kind": page.task, "data": data}


def _reason(error: BaseException) -> str:
    """Why this page failed, in a form the platform will still store as words.

    It redacts URLs, paths, and anything token-shaped out of a callback error
    before storing it, so a reason built from a traceback's worth of file paths
    arrives as a row of placeholders. Collapsed, truncated, and falling back to
    the exception's own name when it has nothing to say.
    """
    text = " ".join(str(error).split())
    if not text:
        text = error.__class__.__name__
    return text[:MAX_REASON_CHARS]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _report_refusal(errors, refusal: AgentVersionRefused) -> None:
    """A 426 is not a blip to back off from - it is an instruction."""
    errors.print(f"[red]{refusal}[/red]")
    errors.print()
    table = ui.detail_table()
    table.add_row("This agent", ui.value(refusal.agent_version or "not stated"))
    table.add_row("Platform needs", ui.value(f"{refusal.minimum_version} or newer"))
    table.add_row("Newest release", ui.value(refusal.latest_version))
    errors.print(table)
    errors.print()
    errors.print(f"Upgrade with [bold]{refusal.upgrade_command}[/bold], then run this again.")


def _report_outdated(console, notice: AgentNotice) -> None:
    """Said once per session, not once per claim: the notice rides every claim
    response, and repeating it every few seconds would bury the work."""
    console.print(
        f"[yellow]This agent is {notice.agent_version}; {notice.latest_version} is out.[/yellow] "
        f"It is still being served. Upgrade with [bold]{notice.upgrade_command}[/bold]."
    )
    console.print()


def _summarise(console, done: int, failed: int, headline: str) -> None:
    console.print()
    console.print(f"{headline} {_count(done, 'page')} done, {failed} failed.")


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


def _bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _short(identifier: str) -> str:
    return identifier.split("-", 1)[0]
