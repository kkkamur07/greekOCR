# Frontend Content-Security-Policy

Applies to the SPA at `app.nomikos.app`. The policy is served as a static
header from [`nomikos/frontend/vercel.json`](../../nomikos/frontend/vercel.json).

## `connect-src` names no loopback origin at all

The policy grants exactly two origins:

```
connect-src 'self' https://api.nomikos.app
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
`next build` at `nomikos/frontend/.next/server/app/index.html` contains five
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

Move the policy from `vercel.json` into `nomikos/frontend/middleware.ts`:
generate 16 random bytes per request, set
`script-src 'self' 'nonce-<value>' 'strict-dynamic'`, and forward the nonce on
the request headers so Next.js can apply it. `output: "standalone"` means
middleware runs, so this is available - it was not done here because it changes
the response path for every request on the app and needs browser verification of
hydration, which belongs with a frontend change rather than a header change.

Until then, `'unsafe-inline'` in `script-src` is an accepted, documented risk.
`object-src 'none'`, `base-uri 'self'`, and `frame-ancestors 'none'` remain in
place and limit what an injected inline script can escalate to.

## `frame-src 'self' blob:` - why the inline PDF preview needs it

The transcription PDF preview
([`PageEditorTranscriptionPdfPane`](../../nomikos/frontend/src/components/page-editor/PageEditorTranscriptionPdfPane.tsx)
and [`PublicCanvasPdfView`](../../nomikos/frontend/src/components/public/PublicCanvasPdfView.tsx))
fetches the PDF over the API, wraps the response in `URL.createObjectURL`, and
embeds the resulting `blob:` URL. Both components used to embed it with
`<object data={blobUrl} type="application/pdf">`, which `object-src 'none'`
blocks outright. The pane rendered empty.

This was previously filed as a Safari quirk. It is not: `object-src 'none'` is
honoured by every CSP-conforming browser, so the preview was blank everywhere the
header was served. It was not caught earlier because the local dev server does
not serve `vercel.json`'s headers - only the deployed app does.

Two things changed:

1. Both components now embed the PDF with `<iframe src={blobUrl}>` instead of
   `<object>`. `object-src 'none'` stays exactly as it was; nothing re-enables
   plugin content. Re-permitting `<object>` would have been the larger
   concession, because `<object>`/`<embed>` is the classic sink for
   plugin-handled content types and `object-src` is the only directive that
   governs it.
2. The policy gained `frame-src 'self' blob:`.

`blob:` has to be named explicitly. It is a distinct scheme, so it is **not**
covered by `'self'` and it is **not** covered by the `default-src 'self'`
fallback that `frame-src` would otherwise inherit - the same reason this policy
already spells it out in `img-src 'self' data: blob:` and `worker-src 'self'
blob:`. An `<iframe>` pointed at a `blob:` URL under the old policy was blocked
just as the `<object>` was.

### What this permits, honestly

`frame-src 'self' blob:` lets the page frame two things: documents from the app's
own origin, and blob URLs. A blob URL only exists because script running on this
origin called `URL.createObjectURL`, and it is scoped to that origin - it cannot
name a remote document, and no third-party origin becomes frameable. So the
grant does not widen the set of *remote* content the app can pull in; it widens
the set of *locally minted* documents the app can frame.

The residual risk is real but bounded, and it is downstream of the
`'unsafe-inline'` gap above rather than independent of it: script that is already
executing on this origin can now mint a blob and frame it, which gives it a
same-origin document to render attacker-chosen markup into. Script that has
reached that point can already write into the live DOM, so this is not a new
capability so much as a second route to one it holds. `frame-ancestors 'none'`
still prevents anyone framing *us*, and `X-Frame-Options: DENY` backs it up.

The alternative - rendering the PDF with PDF.js and no frame at all - was
rejected as disproportionate: `pdfjs-dist` is not a dependency, and adding a full
PDF renderer to remove one narrowly-scoped directive is a worse trade than the
directive.

### The embed can still fail, so it is no longer the only way out

Neither `<object>` fallback children nor `<iframe>` fallback content fires
reliably when a blob embed is refused, which is why the original symptom was a
blank pane rather than the fallback text both components had written for exactly
this case. Both components now render an always-visible "open in a new tab"
link - and, in the editor pane, the existing Download button - *outside* the
embed, so a user is never dependent on the embed reporting its own failure. A
top-level navigation to a `blob:` URL is not governed by `frame-src` or
`object-src`, so that route stays open even if a browser refuses the frame.
