# Rate limiting and client attribution

Code: [`nomikos/backend/users/api/rate_limit.py`](../../nomikos/backend/users/api/rate_limit.py),
[`nomikos/backend/core/api/client_failures.py`](../../nomikos/backend/core/api/client_failures.py).

## The problem: the TCP peer is not the client

`request.client.host` is whoever opened the socket. On the production platform
API - Vercel serverless functions - that is the platform's proxy tier, not the
browser. `client_ip_for_request` only substitutes a forwarded header when
`BEHIND_PROXY=true` **and** the peer falls inside `FORWARDED_ALLOW_IPS`, and the
production deploy sets `BEHIND_PROXY=false` with `FORWARDED_ALLOW_IPS` unset
(see [`docs/deployment/production.md`](../deployment/production.md)). So every
request in production hashed to the same handful of peer addresses.

That is worse than no limit. A bucket of 10 attempts per 60 seconds shared by
the entire user base is not a defence against an attacker - who simply uses the
same budget everyone else is competing for - and it *is* a denial of service
against legitimate users, who get 429s from other people's traffic.

### Why this was not "fixed" by allowlisting Vercel's proxies

`BEHIND_PROXY=true` requires `FORWARDED_ALLOW_IPS` to name explicit addresses or
CIDRs (`AppSettings._require_proxy_allowlist` rejects `*`). Turning it on
without a verified, stable source range is strictly worse than leaving it off:
the peer check would fail, the code would fall back to the peer anyway, and if
the allowlist were ever made loose enough to match, `X-Forwarded-For` becomes
attacker-controlled and the limiter becomes trivially bypassable by spoofing one
header.

**No trusted-proxy configuration is claimed here.** Establishing one means
observing, from a live deployment, the actual peer address a function sees and
confirming the platform overwrites rather than appends the forwarded header.
That has not been done, and asserting it in a config file without the
measurement would be the same mistake in a new place. Until it is done, the
address is treated as unattributable.

## What is enforced instead

### `TRUST_PEER_IP`

`AppSettings.trust_peer_ip` (env `TRUST_PEER_IP`, default `true`) declares
whether the peer address identifies one client. Docker, a VM, or any deployment
reached directly leaves it `true`. Vercel sets it to `false`.

When it is `false` and no trusted forwarded header is configured,
`attributable_client_ip()` returns `None` and **IP-keyed buckets are skipped
entirely** rather than collapsed into one global bucket. Skipping is the correct
failure mode: a global login limit is a self-inflicted outage.

### Per-account throttling on auth routes

`/auth/login` and `/auth/register` are now charged against two independent
buckets:

| Key | When it applies | What it caps |
|-----|-----------------|--------------|
| `ip:<addr>:<path>` | only when the address is attributable | one client's attempt rate |
| `account:<sha256(email)>:<path>` | whenever the body names an account | online guessing against one account |

The account bucket does not care about network topology, which is exactly why it
was chosen: it is the control that still works when the peer address is
meaningless. The email is hashed so the limiter table never becomes a list of
registered addresses. Both buckets are checked before either is charged, so a
request rejected by one does not consume the other's budget.

#### The account is read from the body, never from the headers

Because the account bucket is the only bucket in production, anything that stops
it from being derived removes the limit entirely. Identity extraction therefore
ignores what the request *says* it is sending:

- Media types are case-insensitive (RFC 9110 §8.3.1). A gate comparing
  `Content-Type` against `"application/json"` byte for byte let
  `Content-Type: Application/JSON` past while FastAPI parsed the body and checked
  the password anyway. Same for `application/vnd.api+json` and for a request that
  declares no content type at all, since FastAPI parses both.
- A body too large to attribute is refused, not skipped. Pydantic ignores
  unknown keys, so a login payload padded past `MAX_IDENTITY_BODY_BYTES` with a
  junk field still authenticates. Such a request gets a 413, because "we cannot
  read who this targets" must not resolve to "so charge it to nothing".

The rule is that the bytes decide. Anything that is not JSON cannot reach a
password check either, because every route under this dependency binds a pydantic
model, which FastAPI only fills from JSON, so there is nothing else to probe.

### Requests with no attributable dimension at all

An empty key list does not mean the request goes through unmetered. It is
charged against a coarse, per-path `unattributable:<path>` bucket
(`UNATTRIBUTABLE_AUTH_RATE_LIMIT`, 300 per window).

A shared bucket is safe here for the reason it would be an outage on the main
path: nothing that lands in it is a sign-in. A body with no `email` never reaches
password verification, so a legitimate user cannot be locked out of a bucket they
never enter. It is not a per-client limit and does not pretend to be one. It
bounds what an unattributable caller can make the database do for free.

What this does not stop is credential stuffing spread across many distinct
accounts from a single unattributable source. Nothing keyed on request content
can, and nothing keyed on the peer address can either while the peer is shared.
Closing that gap requires either a verified trusted-proxy configuration (above)
or a platform-level WAF rule. It is an open risk, not a solved one.

### Device pairing has its own throttle, not `throttle_auth_attempts`

`POST /device/v1/pairings` (`nomikos/backend/ml/api/device_pairing.py`) used to
sit under `throttle_auth_attempts`. Its body carries no `email`, so it had no
account dimension to fall back on: every honest pairing start shared the one
`unattributable:/device/v1/pairings` bucket, and one attacker filling that
bucket locked every researcher out of `nomikos pair`.

It is now charged against a dedicated dependency,
`throttle_device_pairing_starts` (`nomikos/backend/users/api/rate_limit.py`).
That dependency keys on `attributable_client_ip(request)`: when the address
identifies one client, it charges a per-client bucket
(`device-pairing:<addr>`, `device_pairing_rate_limit_requests` per window);
when the address is not attributable (e.g. `TRUST_PEER_IP=false` with no
trusted forwarded header), nothing is charged, for the same reason IP-keyed
buckets are skipped elsewhere in this module - a global bucket on this route
is the outage. The backstop against an unattributable flood is the
platform-wide live-pairing ceiling in `DevicePairingService.start_pairing`,
not a rate-limit bucket.

`POST /device/v1/pairings/token` remains deliberately outside
`throttle_auth_attempts` as well; see its own docstring for why (poll cadence
is enforced on the pairing row instead).

## Client-failure beacon

`POST /client-failures` is unauthenticated. It used a per-process in-memory
`dict` of `deque`s, which is the pattern `rate_limit.py`'s own module docstring
rejects: it divides the limit by the worker count, and on serverless it resets
on every cold start, so it enforced nothing.

It now shares the Postgres-backed store:

- attributable address: 30 reports per 60 seconds per address;
- otherwise: 300 reports per 60 seconds, shared.

A shared ceiling is acceptable *here* and not on login, because the endpoint only
writes a log line. Shedding excess beacons degrades observability under flood -
which is the thing the cap exists to protect - and no user-facing flow depends
on the response.

## Residual risks

- Every throttled request performs a small transaction (advisory lock, delete,
  count) before it can be rejected, so a flood still costs database work. The
  advisory lock serialises same-key traffic, which bounds concurrency but not
  arrival rate. An edge/WAF rule is the right layer for volumetric defence.
- `auth_rate_limit_attempts` rows accumulate one per attempt per key within a
  window; expired rows are deleted lazily, on the next request touching the same
  key. A key that is never hit again keeps its rows until something sweeps them.
