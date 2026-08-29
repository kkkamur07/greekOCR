# 0002. Local inference is a CLI, not a loopback service

- Status: Accepted
- Date: 2026-08-04
- Builds on: [0001](./0001-outbound-helper-device-pairing.md), which inverted the transport.
  This record decides what the local process *is* once nothing calls into it.

## Context

ADR 0001 established that the helper stops listening and starts calling. It did not decide
whether the loopback listener survives alongside the outbound path, or what shape the local
process takes once the browser no longer talks to it.

Two things were true and unresolved:

- Roughly 2,500 lines exist solely because a hosted page called `127.0.0.1:8001`. That is the
  frontend discovery, probe and client layer, the editor's local-run UI,
  `local_inference_service.py`, and the helper's own FastAPI server, CORS allowlist, admission
  middleware, and `connect-src` entry.
- Roughly 1,137 lines of per-OS packaging (PyInstaller spec, excludes list, three installer
  script sets, bundle verifier) wrapped 516 lines of program, plus a four-way CI matrix,
  Developer ID notarization, and Authenticode signing.

## Decision

Delete the loopback path outright, and ship local inference as a CLI installed from PyPI. No
dual-mode transition, no native installers, no background daemon, no autostart.

### Nothing bad happens when it is not running

This is the load-bearing property, and every other decision here depends on it. The execution
target is fixed at submission, and may only be chosen when that host has capacity, meaning a
device for it was seen recently. The researcher is always told which host will run the job. A
job never silently changes host, and a job never waits on a machine that is not there.

That makes "not running" an ordinary, announced state rather than a failure, which is precisely
what removes the need for a daemon, and with it the installers. There is no cloud-fallback
timer, no hold window, and no sweeper for unclaimed local jobs, because the decision is made
once, before the job exists.

Because ADR 0003 makes a hosted worker a device like any other, capacity is one question with
one answer rather than two. Cloud capacity will not exist for some time, and that requires no
special handling: submission finds no capacity on either host and says so, rather than creating
a job no one will claim. When hosted workers are eventually stood up they simply begin
answering the same query.

### Choosing the target

An account-level setting ("use my computer when it is available") plus implicit selection.
There is no per-job toggle. A researcher cannot know which host is faster for a given page, so
asking at every action is a decision without a basis, and it is exactly what regrows the
three-mode complexity this record deletes.

The announcement line is therefore not cosmetic. It is the entire user interface for this
feature, and it belongs on the job rather than in a toast.

### Consequences for ADR 0001

Decision 5's stated constraint, "zero terminal use is the product constraint", is retired. It
was written for an invisible daemon, which had to be always-on because a missing helper
silently broke local inference. Once absence is announced and routed, the constraint protects
nobody.

The decisions built on it survive, and only their rationale changes. One improves: decision 13
requires the helper to display the `confirmation_code` and notes that nothing does yet. In a
CLI that is a `print()`, so the anti-phishing mitigation stops being decoration.
`webbrowser.open()` likewise drops from *the only affordance the process has* to a convenience,
since the URL is printed first and always, because `webbrowser.open()` is actively wrong over
SSH.

`test_deployment_hardening.py:71` (`pystray` must not appear) now guards a constraint we no
longer hold, and should stay anyway, because a CLI has no business growing a tray icon.

## Decisions and rationale

### One published package, not two

`nomikos-inference` carries the library and the CLI entry point. The hosted worker installs
the same package. Two packages would buy a version-compatibility matrix between components that
always ship together.

The boundary moves `hf` weight resolution *in*, since it is already on the runtime path
(`adapter.py`, `weights/__init__.py`, `run_errors.py`, both BLLA modules) despite living
outside `nomikos_inference/`.

> Superseded in part by [0004](./0004-pytorch-is-the-inference-runtime.md). This section
> originally moved the Torch modules *out* of the published package and treated `excludes.txt`
> and `scripts/verify-bundle.py` as things a real package boundary would make unnecessary. ADR
> 0004 makes PyTorch the inference runtime, so the Torch modules stay *inside* the package and
> both files die by construction, because there is no forbidden dependency left to police. The
> reasoning below was bundle-era: it inherited "keep Torch out" from the frozen installer
> without rechecking whether PyPI distribution still required it. It did not.
>
> As built (issue 050): the wheel is `nomikos_inference/` minus `nomikos_inference/api` and `nomikos_inference/helper`,
> the loopback HTTP surfaces this record deletes, so the published closure carries no web
> server. `hf` resolution moved in as `nomikos_inference/hub`, and the Hub cache moved with it, from
> beside the code to `~/.nomikos/hf/cache`, because inside a wheel "beside the code" is
> site-packages. The repository root is the project, because a build backend cannot reach
> outside its own root and the package it publishes is `nomikos_inference/`.

### Auto-update at launch, from the platform

A CLI has something a daemon does not: a launch moment with no in-flight work. The agent asks
the platform for its version floor on start, self-upgrades and re-execs if it is below it,
prints a notice when merely outdated, and then begins claiming. Never mid-session, because a
process that swaps its own code during a batch is a bug generator.

The signal comes from the platform rather than PyPI, for the same reason ADR 0001 puts every
cadence in `DeviceSettings`: it is turnable without a release, and it gives the claim endpoint
the ability to *refuse* stale agents, which frozen installers made impossible.

Accepted risk: auto-upgrade executes newly fetched code without asking. A compromised PyPI
package reaches every researcher's laptop at next launch. This is mitigable by pinning to
published hashes, but not eliminable.

> As built (issue 058): "asks the platform for its version floor" needed somewhere to ask. The
> claim endpoint could not be it, because an agent that had to claim in order to learn it was
> stale would be holding a page at the exact moment it replaced its own code. So the floor is
> also served on its own, at `GET /device/v1/agent/version`, from the same comparison and with
> the same 426. It is unauthenticated because the version dependency already resolves before
> any credential is looked at, and it discloses nothing a 426 does not.
>
> Two narrowings of the accepted risk, neither of which closes it. The platform names a
> *package* and never a command, so a compromised platform cannot choose what executes on a
> laptop. And the upgrade runs whichever installer already owns the environment (`pip` if the
> interpreter has one, else `uv pip`), so the index is the researcher's configured one rather
> than one the platform picked. Hash pinning is still the mitigation and is still not done.
>
> The check has exactly one call site, in `main()`, before the command is dispatched. That is
> structural rather than disciplined, so no future claim loop has anywhere to call an upgrade
> from.

### One page per claim

A batch is N claims. Work stays seconds to minutes, so the lease covers it with margin and no
heartbeat endpoint is needed. A slept laptop loses one page rather than a document, and
progress is free because jobs complete as they go.

This is why the claim layer needs exactly one new endpoint. Completion, failure, and graceful
Ctrl-C all post the existing validated `JobCallbackRequest`, and abandonment is caught by the
existing stale sweep. `DEVICE_LEASE_SECONDS` is 600, because the global 1800s is right for a
server that does not sleep and wrong for a laptop that does.

### Page images arrive by short-lived signed URL

The claim response carries a pre-signed link to the one object, expiring in roughly 60 seconds.
That is long enough to download a large scan on a bad connection, and it is not tied to the
600s lease, because the agent fetches once immediately after claiming.

An authenticated `GET /device/v1/jobs/{id}/image` was rejected. The production API is
serverless, so streaming manuscript scans through it costs money for nothing, and it would put
a route on the device token that must independently re-derive job ownership. A signed URL *is*
the authorization.

Accepted risk: a bearer token in a URL leaks through logs and crash dumps. Bounded to one
object and one minute.

### Agent only; standalone inference deferred

The CLI claims platform jobs. It does not yet accept local files
(`nomikos transcribe pages/*.jpg`). The package boundary above makes that a thin wrapper over
`run_model()` whenever it is wanted, but shipping it now would force an output-format decision,
since researchers will want ALTO or PageXML rather than our JSON contract, and that is a
separate commitment.

## Consequences

- Every installed `0.1.6` helper becomes inert. No migration path is provided, because the
  install base is negligible and pre-public. Release assets, the installer CI matrix, and the
  signing credentials are deleted outright rather than deprecated.
- Local inference becomes a technical-user feature. A researcher who cannot use a terminal
  keeps the entire product via cloud, and loses only the ability to run models on their own
  machine.
- Security patching changes character. Frozen installers made us the distributor of a vendored
  dependency tree with no update channel, so every CVE in `onnxruntime`, `protobuf`, `Pillow`,
  or `scipy` required a four-platform rebuild, re-sign, and re-notarize, with no way to make
  anyone install it. It is now `uv tool upgrade`, enforced by the version floor.

## Alternatives considered

Keep loopback as a second mode during transition. Rejected, because every feature (cancel,
progress, cache state, error surfaces, job history) would need two implementations permanently,
to smooth a cutover for an install base small enough to ignore.

Keep `local_only` as a third execution target. Rejected, because its headline justification,
that manuscripts never leave the machine, was never true. Page images already live in the
platform's media store, and the browser downloads them from there today. `local_only` only ever
meant *the model runs here*, and it was the one mode that could leave a job with no terminal
outcome. If an institution ever needs compute-location as a contractual guarantee, it comes
back as a real requirement.

Silent cloud fallback after a hold window. Rejected on the owner's instruction, and correctly:
a researcher who cannot tell where their job ran cannot tell why it was slow. Announcing the
downgrade at submission is simpler than a timer and more honest than silence.

A notice telling users to run `uv tool upgrade` themselves. Safer than self-upgrade and
ignorable, which is the problem, because stale agents are exactly the population that ignores
notices.
