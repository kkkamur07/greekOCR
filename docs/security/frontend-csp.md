# Frontend Content-Security-Policy

Applies to the SPA at `app.nomicous.com`. The policy is served as a static
header from [`nomicous/frontend/vercel.json`](../../nomicous/frontend/vercel.json).

## `connect-src` names no loopback origin at all

The policy grants exactly two origins:

```
connect-src 'self' https://api.nomicous.com
```

It used to end with `http://127.0.0.1:8001`, and before that with
`http://127.0.0.1:*` and `http://localhost:*` as well. The wildcards let any
script on the app origin open a connection to **every** service listening on the
visitor's own machine - development servers, database admin UIs, other desktop
apps with an HTTP control port - and an XSS or a malicious dependency on the app
origin inherited that grant. They were narrowed to the single helper port, and
#60 removed the port with the transport (ADR 0002): the browser no longer calls
an **inference agent** at all, so there is nothing for a loopback entry to
permit.

This is the end state the note in this section used to point forward to. Nothing
should put a loopback origin back: an agent reaches the platform outbound, and a
page that needed to reach one would be reintroducing the fragility the redesign
deleted - a hosted HTTPS page calling `127.0.0.1` depends on a browser
permission Chromium gates behind Private Network Access, that other browsers
treat differently, and that any corporate proxy or VPN can break.

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
