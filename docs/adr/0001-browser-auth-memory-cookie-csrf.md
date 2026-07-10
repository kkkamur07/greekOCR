# Browser auth: in-memory access tokens with secure cookie sessions and CSRF

The platform moves off `localStorage` bearer tokens. Short-lived **access tokens**
live in JavaScript memory only. A **refresh/session cookie** (`HttpOnly`,
`Secure`, `SameSite`) on `Domain=.nomicous.com` restores the session after
reload without persisting the access token in the browser. **CSRF protection**
is required on mutating API requests.

Rejected for now: Next.js BFF proxy (cookie only on app domain); long-lived
access tokens in `localStorage`; Supabase Auth.

This matches the P0 security direction in the repository cleanup plan and
unblocks Phase 3d (`next/image`, cacheable media) once `GET /media/*` accepts
the session cookie.
