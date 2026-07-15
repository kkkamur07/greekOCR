# Vercel Preview pairing (frontend ↔ API)

## Goal

PR previews of `frontend-nomicous` should call the matching `api-nomicous`
preview, not production `https://api.nomicous.com`.

## Approach

Use Preview environment variables — no Related Projects / same-origin proxy.

1. **Live PR previews** — `git.deploymentEnabled: true` on frontend + API
   `vercel.json`.
2. **Frontend Preview env** — set `NEXT_PUBLIC_API_BASE_URL` to the branch API
   preview host (e.g. `https://api-nomicous-git-<branch>-….vercel.app`).
3. **CSP** — `next.config.ts` builds `connect-src` / `img-src` from
   `NEXT_PUBLIC_API_BASE_URL`, so Preview does not stay locked to production.
4. **API Preview env** — keep `BEHIND_PROXY=false` (or set
   `FORWARDED_ALLOW_IPS` if you intentionally enable it). Add the frontend
   Preview origin to `CORS_ORIGINS`.

## Dashboard checklist (Preview)

| Project | Variable | Preview value |
|---------|----------|---------------|
| frontend-nomicous | `NEXT_PUBLIC_API_BASE_URL` | API branch preview URL |
| frontend-nomicous | `NEXT_PUBLIC_CSRF_COOKIE_NAME` | same as Production |
| api-nomicous | `BEHIND_PROXY` | `false` |
| api-nomicous | `CORS_ORIGINS` | include frontend Preview origin |

Production `NEXT_PUBLIC_API_BASE_URL=https://api.nomicous.com` stays unchanged.

## Cookie caveat

`vercel.app` is a public suffix, so two Preview hosts are cross-site. Session
cookies with `SameSite=Lax` will not be sent on browser → API calls the way
they are on custom production domains (`*.nomicous.com`). Prefer short-lived
access tokens for Preview auth smoke tests, or log in again if refresh fails.
