# 0001. Outbound helper device pairing and device-scoped tokens

- Status: Accepted
- Date: 2026-08-03
- Scope of this record: the platform-side device layer, meaning the `helper_devices` and
  `helper_pairings` tables, the pairing endpoints, and the device-token authentication
  dependency. The claim and lease protocol that sits on top is described here only where it
  constrains the device layer, and it ships separately.

## Context

The inference helper is a PyInstaller-frozen background process on a researcher's laptop.
Today the browser calls it over loopback: an HTTPS page at `app.nomicous.com` issuing
`fetch("http://127.0.0.1:8001/...")`.

That is the worst transport the web platform offers. It has to be won on four independently
breakable fronts at once: a CORS allowlist, Chromium's `allow_private_network`, Chromium's
`targetAddressSpace` local-network access gate, and a hand-maintained `connect-src` in
`nomicous/frontend/vercel.json`. Only Chromium passes all four. Safari and Firefox detect the
helper with a simple `GET` and then fail every preflighted `POST`, which is worse than not
detecting it, because the UI promises local inference and then breaks.

Worse, the local path re-implements a pipeline the platform already has and already runs
correctly for cloud jobs:

```
JobSubmitRequest -> run -> JobCallbackRequest -> JobCallbackService.apply_callback
  -> notify_platform_job_status_changed -> SSE -> jobSubscription.ts
```

All five stages exist. The loopback path duplicates all five, badly, in a browser tab, where a
closed tab or a sleeping laptop loses the job.

## Decision

Invert the direction. The helper stops listening and starts calling. It pairs once against
`api.nomicous.com`, then long-polls a claim endpoint, runs the job, and posts the ordinary
`JobCallbackRequest` back. The browser never touches loopback, so the entire four-front
transport problem is deleted rather than patched.

Pairing follows RFC 8628 (device authorization grant) with the typable `user_code` removed,
because our device can open a browser and a television cannot.

### The offline question, asked and answered

Requiring network connectivity to *receive* work is a real change, since today the helper needs
no network at all beyond the browser on the same machine.

The question was put to the owner directly, and the answer is that researchers almost always
have access to `api.nomicous.com`. Fully-offline archive work is not a workflow this product
serves today. The outbound claim loop is therefore viable, and this is settled. It is recorded
here so it does not get re-litigated, not so it can be reopened.

What we accept in exchange: a laptop with no network cannot pick up work. What we gain: the job
survives a closed tab, a sleeping laptop, an uninstalled helper, and a browser that is not
Chromium.

## Decisions and rationale

### 1. No typable code at all

RFC 8628 needs a `user_code` because a TV cannot open a browser. Ours can. `webbrowser.open()`
is available on every platform the helper ships to, and it is the only affordance the helper
has, since it is `LSUIElement=true`, has no dock icon and no window, and `pystray` was
deliberately removed and is asserted against at
`tests/nomicous/unit/test_deployment_hardening.py`.

A 6 to 8 character typable code carries 30 to 40 bits and is the only brute-forceable surface
such a protocol has. Not having one beats defending one. Both secrets are
`secrets.token_urlsafe(32)`, which is 256 bits.

Justifying that size against lifetime and rate limit:

| Secret                | Entropy  | Lifetime                                                            | Guess budget                                                                                                          |
| --------------------- | -------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `verification_token`  | 256 bits | 300 s, extendable by *successful* polls to a 900 s hard cap         | throttled at 10 req/60 s on the routes that touch it; roughly 2^255 expected requests                                 |
| `device_code`         | 256 bits | same                                                                | caller must also present the matching `pairing_id` (122-bit UUID4); 5 wrong presentations burn the row permanently    |

Closed-tab recovery, the one thing a typable code is genuinely good for, is served by starting
a new pairing from the helper. It was originally served by `GET /devices/pairings`, an IP-scoped
list of live pairing requests. That route has been deleted; see "The client IP is not a real
identifier" below.

There is now one short string in the protocol, the confirmation code, and it is deliberately
not a `user_code` under another name: no endpoint accepts it, so it adds no brute-forceable
surface. It is a keyed derivation of the `pairing_id`, shown by the helper and on the consent
screen so a human can compare them. See decision 13.

### 2. The browser handoff token travels in the URL fragment

`verification_url` is `https://<app>/pair#<verification_token>`. A query string lands in the
frontend's Vercel access logs, in browser history, and, absent a strict `Referrer-Policy`, in a
`Referer` header. `location.hash` reaches none of those, and after `history.replaceState` it is
gone from the address bar too.

It is then submitted to the API in a POST body (`POST /devices/pairings/lookup`), never in a
path or a query, so it never appears in a server-side access log either.

### 3. Reuse the existing opaque-credential scheme, do not invent a second one

`backend/users/application/browser_sessions.py` already solves "store an opaque credential":
`hmac.new(key, secret, sha256).hexdigest()` into a `String(64)` column, verified with
`hmac.compare_digest`, with the row id embedded in the wire value so lookup is a primary-key
fetch. `backend/ml/application/opaque_tokens.py` is the same scheme, and a unit test asserts
byte-for-byte agreement with `browser_sessions._hash`.

bcrypt is deliberately not used. It is the password hash (`users/application/password.py`, 12
rounds), a slow KDF for low-entropy human input. Running it over a 256-bit random string buys
nothing and costs roughly 100 ms on a path a helper hits on every renewal.

There is no index on `token_hash`, because we never search by digest.

### 4. A dedicated `X-Nomicous-Device-Token` header, not `Authorization: Bearer`

`get_current_user` runs `HTTPBearer` then `decode_access_token`, so a device token in
`Authorization` would already fail JWT decode. A separate header removes the question entirely,
and matches the existing `X-Inference-Webhook-Secret` idiom in
`backend/jobs/api/dependencies.py`.

`get_current_device` also returns a different type (`AuthenticatedDevice`, not `User`), so no
existing route can silently start accepting a device token by changing an annotation.

### 5. A dedicated `DEVICE_TOKEN_HMAC_SECRET`, and production refuses to start without it

Rotating `JWT_SECRET` today logs browsers out. That is recoverable, because a researcher logs
in again. It must not *also* silently unpair every UI-less laptop, which is not recoverable
without a terminal, and zero terminal use is the product constraint.

The first cut made this a silent fallback to `JWT_SECRET`, and the variable appeared nowhere
outside its own settings module, the tests, and this record. That is the same failure as not
having the setting at all: every production deployment would have keyed device tokens off
`JWT_SECRET` and nobody would have known until a routine rotation unpaired every laptop. So:

- `DEVICE_TOKEN_HMAC_SECRET` is in `.env.compose.example` and
  `nomicous/backend/core/.env.production.example`, with the blast radius spelled out next to
  it;
- in production, `DeviceSettings` refuses to construct when the secret is unset or equal to
  `JWT_SECRET` *and* pairing is enabled. `create_app()` resolves it before mounting anything,
  matching how `JWT_SECRET` and the inference secrets already fail fast;
- while pairing is disabled it logs a warning instead, so a deployment that has not turned the
  feature on is not blocked by a secret it does not yet use;
- the fallback survives outside production, where a shared key costs nothing.

### 6. Two tables, not one

Folding pairings into `helper_devices` would force `user_id` to be nullable and destroy the
database-level invariant that a device belongs to exactly one researcher. Every device query
would then need `AND approved_at IS NOT NULL`, and one forgotten clause is an authentication
bypass. `helper_devices.user_id` is `NOT NULL REFERENCES users(id) ON DELETE CASCADE`, and that
foreign key *is* the entire authorization scope of the credential.

### 7. Strict single-use pair codes (deviation from the reviewed design)

The reviewed design allowed a consumed pairing to re-mint up to three times within five
minutes, so a helper that lost the response could recover.

We ship strict single-use: once `consumed_at` is set, every later poll returns `access_denied`.
The trade:

- The lost-response case is cheap to recover from. The helper simply starts a new pairing and
  re-opens the browser, the researcher's tab is still open, and the IP-scoped pairing list finds
  it. Cost: one extra click.
- The re-mint window is not cheap. A `device_code` captured after the legitimate helper has
  already collected its token would still redeem, handing an attacker a valid, long-lived token
  for that researcher's account, with the legitimate helper working normally so nothing looks
  wrong.

`delivery_count` is kept on the row so the window can be reintroduced if the field demands it,
but it ships at one.

### 8. Approval re-verifies the `verification_token`

The reviewed design's approve endpoint took only a `pairing_id`. We require the
`verification_token` in the body and verify it constant-time against the row, so possession of
the fragment, rather than knowledge of a pairing id, is what authorises the grant. The `/pair`
page already holds the token in memory, so this costs nothing in UX.

### 9. The pairing poll is not under `throttle_auth_attempts`

That limiter is 10 requests / 60 s keyed on `{ip}:{path}` (`users/api/rate_limit.py`,
`settings/auth.py`). A compliant 5 second poll is 12 requests per minute, so a well-behaved
helper would throttle *itself* at poll 11.

Cadence is enforced on the pairing row instead. Polling sooner than `interval_seconds - 1`
returns `slow_down` with the interval doubled, capped at 30 s, and does not consume an attempt.
`POST /device/v1/pairings`, called once per installation, is under the shared limiter, where it
belongs.

### 10. Protocol states ride in 200 bodies

`backend/core/app.py` discards `HTTPException.detail` and replaces it with a fixed public
string per status code. Only `ConflictError` (409) passes a readable message through. A
machine-readable protocol state therefore cannot survive a non-2xx response, so
`POST /device/v1/pairings/token` always returns 200 with
`status ∈ {authorization_pending, slow_down, access_denied, expired, approved}`.

### 11. Revocation is a row read, not a cache expiry

Every device request re-reads `helper_devices`, with no JWT, no cache, and no TTL, so
`DELETE /devices/{id}` lands on that device's very next call. Revocation also blanks
`token_hash`, so the credential is dead even if `revoked_at` were later cleared. It works from a
phone and needs no cooperation from the helper, no local network, and no loopback.

### 12. No per-request rotation; a 180-day TTL with an explicit renewal overlap

A browser that loses a rotation response logs in again. A UI-less helper would be bricked with
no way to tell anyone. So `token_expires_at = issued + 180 d`, with an explicit
`POST /device/v1/token/renew`, and the predecessor stays valid for
`DEVICE_TOKEN_RENEW_OVERLAP_HOURS` (24 h) in `previous_token_hash`. A lost renewal response is
harmless.

### 13. A confirmation code, because the consent screen had nothing checkable on it

Everything the consent screen showed was supplied by whoever started the pairing: `device_name`,
`platform`, `helper_version`. An attacker sets those to `"MacBook Pro - Nomicous Helper"`,
`"darwin-arm64"`, `"0.2.0"` and the screen looks exactly like the honest case. The one field
that was not attacker-supplied, `same_network`, was inert (see below). So the screen asked for
consent and gave the researcher nothing to base it on.

`confirmation_code(pairing_id, hmac_key)` is a keyed derivation rendered as two groups of four
characters from a 32-symbol alphabet with `I`, `O`, `0`, and `1` removed. It is returned to the
helper at `POST /device/v1/pairings` and shown on the consent screen. It is not a secret: it
opens nothing, it is derived rather than stored, and an attacker who starts a pairing learns
their own code. The point is that the victim's screen then shows the *attacker's* code, which
the victim's own helper never displayed.

This works only if two things happen outside this layer. The helper must display or log it, and
the `/pair` page must show it prominently enough to be compared. Neither exists yet, and both
are stated as requirements below.

### 14. The client IP is not a real identifier here, so nothing is keyed on it

`client_ip_for_request` returns the direct peer unless `BEHIND_PROXY=true` and that peer matches
`FORWARDED_ALLOW_IPS`. The Vercel deployment sets `BEHIND_PROXY=false` because no stable
allowlistable source range is available for it. Every request therefore presents the same
address, the edge's.

Three controls were built on that address, and all three inverted:

1. `GET /devices/pairings` filtered on `request_ip` and nothing else, by construction, because a
   pairing has no owner before consent. With one shared address the filter matches every row, so
   every authenticated user would have seen every other user's live pairing requests,
   `pairing_id` included. Deleted. Closed-tab recovery costs one click in the helper instead.
2. The per-IP live-pairing cap of 3 became a platform-wide cap. Three unauthenticated requests
   would have blocked pairing for everyone for the pairing lifetime. Replaced by
   `DEVICE_PAIRING_MAX_LIVE_TOTAL`, an explicit global ceiling with a default of 10 000, paired
   with a sweep of finished rows. The honest description of that number: it bounds the table,
   and it does not stop an adversary. Anyone who can sustain a flood against the route can hold
   it full, but that is a request flood, indistinguishable from a flood against any other
   endpoint, and it is answered at the edge rather than by a counter in this table. The
   three-row cap was different in kind, because it made *three* requests sufficient.
3. `same_network` was unconditionally `true`. An anti-phishing signal that always reads "safe"
   is worse than no signal, because the screen presents it as evidence. Removed, along with
   `request_ip`, from the consent DTO. Both columns are still written for support correlation,
   and neither is shown to a researcher as though it described their computer.

The alternative, allowlisting Vercel's egress ranges and setting `BEHIND_PROXY=true`, was not
taken, because it requires *proving* the range is fixed and exclusive to this deployment, and an
allowlist that is merely plausible turns `X-Forwarded-For` into a client-controlled identity. If
that range is ever established and verified, these controls can come back. Until then the honest
position is that this platform does not know its callers' addresses.

This cuts wider than the device layer. `throttle_auth_attempts` is keyed on the same value, so
`AUTH_RATE_LIMIT_REQUESTS` is a global budget for `/auth/login` and friends rather than a
per-caller one. That is the same root cause and is being addressed in the rate-limiting layer,
not here. The device layer deliberately does not *depend* on that throttle for any security
property, so whichever way it lands, nothing above changes.

## Consequences

### The new risk, stated plainly

A long-lived credential now sits on a researcher's laptop where none existed. This is the
single largest change in the security posture of the product and it is not fully mitigable.

Whoever can read the helper's credential file can claim that user's local-eligible jobs and read
the page images for exactly those jobs, until revocation. Bounded by:

- per-user scope as a `NOT NULL` foreign key, rather than a code-review promise;
- no access to any resource that was not handed over as a claimed job: no documents, no
  projects, no job enumeration, no other user;
- revocation within one poll cycle, with no cache to wait out;
- `0600` in a `0700` directory on the client, never in the LaunchAgent plist
  (`~/Library/LaunchAgents` is world-readable and lands in Time Machine) and never in the
  environment;
- `token_prefix`, `paired_from_ip`, and `last_seen_ip` in the UI for log-safe support
  correlation.

Detection is honestly weak, because a stolen token looks exactly like a second legitimate
laptop. Rotation does not help, since the threat is a readable file on a machine the attacker
already controls. Only revocation helps.

### Pairing phishing, the residual risk, not mitigated away

The consent link is transferable, and this layer cannot make it otherwise.

The attack, concretely and without softening: someone with no account on the platform POSTs
`/device/v1/pairings` with `device_name = "MacBook Pro - Nomicous Helper"`, gets back
`https://app.nomicous.com/pair#<verification_token>`, and emails that link to a researcher. If
the researcher is logged in and clicks approve, the platform mints a 180-day device token bound
to the researcher's account, for the attacker's process. That credential can claim that
researcher's local-eligible jobs and read the page images for them, until someone revokes it.

Nothing in the protocol binds an approval to the device that requested it. The approving browser
and the polling helper are two different processes that share only a `pairing_id`, and the
platform sees no channel between them it can verify. Origin, referrer, and IP are all either
absent (fragment token, POST body) or untrustworthy (see decision 14). We have not eliminated
this, and it cannot be eliminated inside the backend.

What this change does buy:

- A confirmation code (decision 13), the first thing on the consent screen a researcher can
  actually check, provided the helper shows it and they compare.
- A much shorter window: 300 s TTL and a 900 s hard cap, down from 900 s and 24 h. A phished
  link now dies in minutes.
- A record of what was approved. `helper_pairings` keeps `approved_user_id`, `approved_at`,
  `device_id`, and the requested strings, and `device_pairing_approved` is logged with the user,
  the device, the presented name, and the confirmation code. Post-incident, "what exactly did I
  click" is answerable.
- Rate limiting on unauthenticated pairing creation, through the shared auth throttle, whose
  per-caller precision depends on the IP question in decision 14.
- `GET /devices` still surfaces the unexpected entry afterwards, and revocation lands on the
  device's next request.

What it does not buy: any of the above stops a researcher who is in a hurry, trusts the email,
and clicks. Treat pairing consent as being in the same risk class as an OAuth consent screen,
and expect the same failure rate.

#### Requirements on the `/pair` page (it does not exist yet)

`DEVICE_PAIRING_ENABLED` defaults to off in production and must stay off until all of these
hold. They are not styling notes, because the backend's mitigations are inert without them.

1. Never approve from page load. Approval requires an explicit click on a button whose label
   names the consequence. No auto-submit, no approve-on-mount, no confirmation in a query
   parameter.
2. Show the confirmation code first and largest. Directly beside it, in plain words: *"Your
   helper is showing a code. If it does not match this one, do not continue."* If the codes
   cannot be compared, the code is decoration.
3. State what is being granted, in the researcher's terms: this computer will be able to receive
   your manuscript pages and run OCR on them, for up to 180 days, until you remove it in
   Settings.
4. Render `device_name`, `platform`, and `helper_version` as inert plain text under fixed
   labels. They come from an unauthenticated endpoint. Never interpolate them into markup, a
   link target, or a heading that could be mistaken for platform copy.
5. Say plainly that these strings are self-reported by the computer asking for access, so a
   convincing name is not evidence.
6. Do not restore any network claim. No "same network", no IP, and no location, because the
   platform does not know the caller's address (decision 14).
7. Clear the fragment with `history.replaceState` after reading it, and keep the token in memory
   only.
8. Make deny as prominent as approve, and make deny the outcome of closing the tab, since an
   unapproved pairing simply expires.
9. Link to the device list from the confirmation state, so an approval a researcher regrets is
   one click from revoked.

A malicious process already running as the user on the researcher's own machine can still start
a pairing and open a legitimate-looking tab, with the correct confirmation code, since it can
read whatever the honest helper would display. Accepted, unchanged: that adversary already has
code execution as the user.

### Consent-screen strings are attacker-controlled

`device_name`, `platform`, and `helper_version` come from an unauthenticated endpoint. They are
normalised server-side (NFC, control and format characters stripped, length-capped) and must be
rendered as inert plain text under fixed labels. They must never be interpolated into markup or
used as a link target.

### Requirements on the helper client

The helper must display or log the `confirmation_code` from `POST /device/v1/pairings` before
opening the browser. Writing it to the helper's log file is the minimum, and showing it in a
notification is better. Without it the researcher has nothing to compare the consent screen
against, and decision 13 is decoration.

It must never write the `device_code`, the `verification_token`, or the device token to that
log.

> As built (issue 056). There is no helper and no log file, because ADR 0002 made the client a
> CLI, so the code is printed to the terminal, before the wait and before any
> `webbrowser.open()`. The pairing URL is printed before it too, so a browser that cannot open,
> or opens on the wrong machine over SSH, now costs nothing. `nomicous pair` prints neither the
> `device_code` nor the device token, and the `verification_token` is unavoidably inside the URL
> a researcher has to be shown. The token is written to `~/.nomicous/device.json` at `0600` in a
> `0700` directory, which is the mode this record's consequences section requires.
>
> The `/pair` page requirements below are still unbuilt, and `DEVICE_PAIRING_ENABLED` should
> stay off in production until they are. The client half of decision 13 now holds up its end,
> the screen half does not exist yet, and one without the other compares nothing.

### Operational

- The three device routers are mounted in `create_app()`. The first cut was not, and the
  integration suite hid it by building its own FastAPI app, so the entire phase was unreachable
  with a green suite. Two tests now assert the routes against the real `create_app()` schema.
- `DEVICE_PAIRING_ENABLED` gates the whole device surface as a router-level dependency,
  evaluated per request rather than at mount time, so the routes stay in the OpenAPI schema and
  the switch does not need a code change. It returns 404, because a disabled feature should look
  like one that was never deployed. Default: off in production, on elsewhere.
- `helper_pairings` is written by an unauthenticated endpoint, so it is swept. Rows past
  `DEVICE_PAIRING_RETENTION_SECONDS` beyond their expiry, and consumed or denied rows past that
  age, are deleted from `start_pairing` itself. That is the one endpoint that inserts into the
  table, which makes cleanup proportional to insertion and needs no background loop. The
  production API is serverless and has none to put it in.
- Both models are registered in `infrastructure/models.py`. They previously reached
  `Base.metadata` only transitively through `users/api/dependencies.py`, which meant the next
  `alembic autogenerate` would have emitted `drop_table` for both.
- Migration `005_helper_devices` chains from `004_document_part_dimensions` and has a working
  `downgrade()`. It is written idempotently, so it stays a no-op on a database whose baseline
  already contains the tables. It issues its own `GRANT` to `nomicous_api`, because
  `002_service_roles` grants `ON ALL TABLES`, which is point-in-time, and
  `ALTER DEFAULT PRIVILEGES`, which only covers tables created by the role that ran it. Neither
  is guaranteed to reach a table created here, and the failure mode is a permission error on the
  first pairing request.
- Every state transition is logged (`device_pairing_started`, `_approved`, `_denied`, `_burned`,
  `device_token_issued`, `_renewed`, `device_revoked`, `device_auth_rejected`) with
  `pairing_id`, `device_id`, `user_id`, and `token_prefix`. No raw secret is ever passed to a
  logger, and a test asserts it.
- Every lifetime, cap, and cadence is an environment dial (`DeviceSettings`), turnable without a
  helper release. The helper reads its own poll cadence from the platform rather than compiling
  it in.
- The claim endpoint must not take `Depends(get_db)`. `get_db` pins a pooled connection for the
  whole request, and a 25 s long-poll per idle device would exhaust the pool at roughly fifteen
  devices. `stream_job_events` already avoids `get_db` for exactly this reason.

## Alternatives considered

Keep loopback and fix the four fronts. Rejected, because the four fronts are independent, two of
them are Chromium-only, and one (`connect-src`) is a hand-maintained list in a file that ships
with the frontend. Fixing all four still leaves Safari and Firefox broken.

A typable `user_code` as a fallback path. Rejected, because it reintroduces the only
brute-forceable surface in the protocol, and it asks a non-technical researcher to type eight
characters correctly. The IP-scoped pairing list covers the recovery case it was there for.

Routing state in `jobs.payload` JSONB rather than a typed column. Rejected for the claim layer
that follows: a predicate like `(payload->>'cloud_fallback_at')::timestamptz <= now()` is a
per-row text cast in the cloud worker's hot claim query, is not index-servable, and a single row
with a non-timestamp string there raises and breaks `claim_next_pending_job` for every job on
the platform. A `timestamptz` column cannot do that.

Server-Sent Events or WebSockets instead of long-polling for claims. Rejected, because
`JOB_SSE_NOTIFICATIONS_ENABLED=false` in production precisely because
`platform_job_notification_loop` needs a per-process `asyncpg` LISTEN that a request/response
function cannot hold. Building the helper's lifeline on the one transport already known not to
work there would be perverse. Vercel functions accept no WebSocket upgrade, and a separate
persistent host would give the helper a second base URL to discover, which is the discovery
problem this redesign exists to delete.

## What this record does not decide

The claim, heartbeat, complete, and release endpoints; `jobs.execution_policy` and the local
hold window; lease sweeping; the signed media URL for page images; the `/pair` page; and the
helper's own client. Those land on top of this layer and get their own record if they make
decisions this one does not already imply.
