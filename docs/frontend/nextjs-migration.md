# Next.js migration record

Decisions from a grill-with-docs session (July 2026). The migration from **Vite +
React Router** to **Next.js App Router** is complete; this document is retained as
the implementation record and follow-up checklist.

Domain terms: [`nomicous/CONTEXT.md`](../../nomicous/CONTEXT.md)
Performance work (ship before or in parallel): [`performance-optimization.md`](performance-optimization.md)

---

## Goals

- Same URLs, same pages, same editor/pairing/inference behavior.
- No user-visible regressions on any of the eight current routes.
- Migration is a **framework shell swap** — perf and auth improvements are separate PRs.

## Non-goals (this migration PR)

- Server Components or SSR (full client SPA on day one).
- Auth implementation (Phase 3d — approach chosen, ships after migration).
- `next/image` or cookie-authenticated media (Phase 3d — after auth lands).

---

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Strategy | **Migrate first, optimize after** — behavior parity, then perf tiers |
| 2 | Location | **In-place** — convert `nomicous/frontend/`, delete Vite when done |
| 3 | Route cutover | **Big-bang** — all 8 routes in one PR |
| 4 | Rendering | **Full client SPA** — `'use client'` on every page; providers in root layout |
| 5 | Navigation state | **Drop `location.state`** — login uses `?callbackUrl=`; editor always fetches document (SWR cache added in Phase 3) |
| 6 | Dev port | **Keep 5173** — `next dev -p 5173`; no CORS/docker-compose churn |
| 7 | Verification | **Vitest (27 suites) + manual smoke** — no Playwright for migration |
| 8 | Perf timing | **Ship Phase 0 perf on Vite first** — migration PR is routing/shell only |
| 9 | Auth (Phase 3d) | **In-memory access token + HttpOnly session/refresh cookie + CSRF** — see [Auth architecture](#auth-architecture-phase-3d) |

---

## Current routes → App Router map

| React Router path | Next.js `app/` file |
|-------------------|---------------------|
| `/login` | `app/login/page.tsx` |
| `/register` | `app/register/page.tsx` |
| `/projects` | `app/projects/page.tsx` |
| `/projects/:projectId` | `app/projects/[projectId]/page.tsx` |
| `/projects/:projectId/documents/:documentId` | `app/projects/[projectId]/documents/[documentId]/page.tsx` |
| `/projects/.../parts/:partId` | `app/projects/[projectId]/documents/[documentId]/parts/[partId]/page.tsx` |
| `/public/projects/:projectId/documents/:documentId` | `app/public/projects/[projectId]/documents/[documentId]/page.tsx` |
| `/` | `app/page.tsx` (redirect authed → `/projects`, else `/login`) |

---

## What changes (migration PR)

| Area | Work | Risk |
|------|------|------|
| Scaffold | `next.config.ts`, `app/layout.tsx`, remove Vite/`main.tsx`/`App.tsx` | Low |
| Routing | `react-router-dom` → `next/navigation` (~30 files) | Medium |
| Auth guard | `ProtectedRoute` → client guard component (middleware later, when auth migrates) | Medium |
| Login redirect | `location.state.from` → `searchParams.callbackUrl` | Low |
| Env vars | `VITE_*` → `NEXT_PUBLIC_*` (3 call sites) | Low |
| Providers | Move `ConfigProvider`, `ToastProvider`, `BackgroundJobsProvider`, `BackgroundJobsPanel` into root layout | Low |
| Dev proxy | Vite `/media` proxy → `next.config` `rewrites` to API | Low |
| Deploy | `vercel.json` framework → Next; update `production.md` | Low |
| Docker | `Dockerfile` — `next build` + `next start` or static export + nginx (decide at implement time) | Low |
| Tests | Replace `MemoryRouter` mocks with `next/navigation` mocks | Low–medium |

## What copies unchanged

~90% of `src/`:

- Page editor, canvas, pairing, local inference hooks
- API client (`fetch` + JWT from `localStorage` until Phase 3d; then in-memory token)
- `AuthenticatedImage`, Ant Design UI, image cache (if Phase 0 merged)
- Business logic and Vitest unit tests (with router mock updates)

---

## Root layout sketch

```tsx
// app/layout.tsx — all client providers or a ClientProviders wrapper
'use client';
// ConfigProvider (antd), ToastProvider, BackgroundJobsProvider
// children + BackgroundJobsPanel
```

Protected routes: client component checks `hasAccessToken()` and redirects to
`/login?callbackUrl=...` (same pattern as today, no SSR flash until cookie auth).

---

## Manual smoke path (definition of done)

1. Login / register / logout
2. Projects list → project dashboard → document detail
3. Open part in editor → draw segment → pairing → save
4. Background job polling / SSE still works
5. Public document page loads without auth
6. Navigate back from editor — document list intact
7. Hard refresh on deep-linked editor URL works

---

## Recommended PR order

```
Phase 0 (Vite) — see performance-optimization.md
  PR 1  Backend ETag / Cache-Control
  PR 2  imageCache + AuthenticatedImage + inference hooks
  PR 3  ?w=200 thumbnails + hover prefetch

Phase 1 (this doc)
  PR 4  Next.js migration (big-bang)

Phase 3 — see performance-optimization.md
  SWR, rendering, bundle, auth (when ready)
```

---

## Effort estimate

| Phase | Estimate |
|-------|----------|
| Phase 0 on Vite | 2–4 days (frontend PRs 2–3; backend PR 1 is hours) |
| Phase 1 Next migration | 2–4 days (lower if Phase 0 already merged — mostly plumbing) |
| Phase 3 (excl. auth) | 3–5 days |
| Phase 3d auth | 5–8 days (after Phase 3a–c) |

---

## Auth architecture (Phase 3d)

**Decision:** in-memory access tokens + secure cookie session + CSRF.
ADR: [`docs/adr/0001-browser-auth-memory-cookie-csrf.md`](../adr/0001-browser-auth-memory-cookie-csrf.md)

### Token split

| Token | Storage | Lifetime | Used for |
|-------|---------|----------|----------|
| **Access token** | JavaScript memory only (`AuthProvider` / module) | Short (e.g. 15 min) | `Authorization: Bearer` on API + media `fetch` |
| **Refresh / session** | `HttpOnly; Secure; SameSite=Lax` cookie, `Domain=.nomicous.com` | Longer (e.g. 7–30 days) | `POST /auth/refresh` after reload or 401 |
| **CSRF token** | Readable cookie or login response + header on mutations | Per session | `POST` / `PUT` / `PATCH` / `DELETE` |

Never store the access token in `localStorage`, `sessionStorage`, or non-HttpOnly cookies.

### Request flow

```
App bootstrap (hard refresh)
  → AuthProvider calls POST /auth/refresh (credentials: include)
  → access token into memory
  → render protected routes

API call
  → Authorization: Bearer <memory>
  → credentials: include (cookie sent)
  → mutating requests: X-CSRF-Token (or equivalent)

401 on access token expired
  → refresh once → retry → else redirect /login

Logout
  → POST /auth/logout (clear cookie)
  → clear memory token + clearImageCache()
```

### Backend changes (FastAPI)

- `POST /auth/login`, `/auth/register` — return `access_token` in JSON + `Set-Cookie` refresh
- `POST /auth/refresh` — validate cookie → new `access_token` in JSON
- `POST /auth/logout` — clear cookie
- `get_current_user` — accept **Bearer OR** session cookie (for media `<img>` later)
- CSRF middleware on mutating routes
- CORS: `Access-Control-Allow-Credentials: true`, explicit origins (`https://app.nomicous.com`)

### Frontend changes

- `AuthProvider` at root — holds access token, exposes `getAccessToken()`, bootstrap refresh
- Replace `auth/storage.ts` localStorage with memory + refresh flow
- `api/client.ts` — read token from provider; `credentials: 'include'` on all requests
- Protected routes — wait for bootstrap before redirect (no flash)
- Next middleware (optional capstone) — forward cookie to `/me` server-side for SSR shell later

### Media / Tier 5 unlock

Once `get_current_user` accepts the session cookie on `GET /media/parts/{id}`:

- `AuthenticatedImage` → plain `<img src>` with `credentials: 'include'` or `next/image` loader
- Browser HTTP cache + ETag from Phase 0 PR 1
- In-memory blob cache remains for inference base64 dedup

### Still TBD at implementation time

- Exact access/refresh TTLs
- CSRF mechanism (double-submit cookie vs synchronizer token)
- Cookie names and rotation on refresh

---

## Related docs

- [`performance-optimization.md`](performance-optimization.md) — image cache, thumbnails, Phase 3
- [`deployment/production.md`](../deployment/production.md) — `app.nomicous.com` Vercel project
- [`nomicous/frontend/README.md`](../../nomicous/frontend/README.md) — current Vite setup (update after migration)
