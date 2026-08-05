# Frontend Content-Security-Policy

Applies to the SPA at `app.nomicous.com`. The policy is served as a static
header from [`nomicous/frontend/vercel.json`](../../nomicous/frontend/vercel.json).

## `connect-src` is one loopback origin, not a port range

The policy previously ended with:

```
connect-src 'self' https://api.nomicous.com http://127.0.0.1:8001 http://127.0.0.1:* http://localhost:8001 http://localhost:*
```

`http://127.0.0.1:*` and `http://localhost:*` let any script running on the app
origin open a connection to **every** service listening on the visitor's own
machine - development servers, database admin UIs, other desktop apps with an
HTTP control port. That is a much larger grant than "talk to our helper", and an
XSS or a malicious dependency on the app origin inherits it.

Nothing needed it. `nomicous/frontend/src/inference/constants.ts` builds exactly
one helper URL:

```ts
const DEFAULT_HELPER_BASE_URL = "http://127.0.0.1:8001";
export const HELPER_BASE_URL = configuredHelperBaseUrl || DEFAULT_HELPER_BASE_URL;
```

with a comment stating that discovery "deliberately does not walk a list of
candidate URLs". Production sets `NEXT_PUBLIC_INFERENCE_HELPER_URL=http://127.0.0.1:8001`
(see [`docs/deployment/production.md`](../deployment/production.md)), which is
the default anyway. The wildcards and the `localhost` spellings are gone;
`http://127.0.0.1:8001` stays.

**Note for the helper redesign:** `http://127.0.0.1:8001` is still a live path -
the browser calls it directly today. When the redesign removes the
browser-to-loopback call, delete that entry too and `connect-src` becomes
`'self' https://api.nomicous.com`. Do not delete it before then; the transcription
flow breaks silently (a blocked `fetch` looks exactly like "helper not running").

If someone needs a non-default helper port, the browser-visible consequence is a
CSP violation rather than a working connection. That is deliberate: a per-user
port would have to widen the policy for everyone.

## `script-src` still needs `'unsafe-inline'` - evidence

This is a real, known weakness. It is kept because removing it under the current
delivery model produces a page that renders and then never hydrates, and shipping
that would be worse than the gap.

Next.js 16 (App Router, `output: "standalone"`) emits inline `<script>` blocks
that carry the React Server Components flight payload. Built output from
`next build` at `nomicous/frontend/.next/server/app/index.html` contains five
inline scripts, none of which has a `nonce` attribute:

```
inline <script> count: 5
  body: (self.__next_f=self.__next_f||[]).push([0])
  body: self.__next_f.push([1,"1:\"$Sreact.fragment\"\n2:I[44636,...
  body: self.__next_f.push([1,"0:{\"P\":null,\"c\":[\"\",\"\"],...
  ...
nonce= present: False
```

Two consequences:

1. **A hash allowlist cannot work.** The payload embeds route- and data-specific
   content, so its digest changes per page and per render. A fixed
   `'sha256-...'` list in `vercel.json` would be stale on the first content
   change. This is the difference from the landing page, which has one static
   JSON-LD block and *is* pinned by hash (see
   `test_landing_csp_uses_json_ld_hash_instead_of_unsafe_inline`).

2. **A nonce cannot come from `vercel.json`.** Nonces must be unique per
   response; a static `headers` entry emits one constant value, which is
   equivalent to `'unsafe-inline'` against an attacker who can read the header.
   Next.js supports nonces only when the CSP header is generated per request in
   `middleware.ts` - the framework reads the nonce out of the request's CSP
   header and stamps it onto its own inline scripts.

### The fix, when it is scheduled

Move the policy from `vercel.json` into `nomicous/frontend/middleware.ts`:
generate 16 random bytes per request, set
`script-src 'self' 'nonce-<value>' 'strict-dynamic'`, and forward the nonce on
the request headers so Next.js can apply it. `output: "standalone"` means
middleware runs, so this is available - it was not done here because it changes
the response path for every request on the app and needs browser verification of
hydration, which belongs with a frontend change rather than a header change.

Until then, `'unsafe-inline'` in `script-src` is an accepted, documented risk.
`object-src 'none'`, `base-uri 'self'`, and `frame-ancestors 'none'` remain in
place and limit what an injected inline script can escalate to.
