# Frontend performance optimization

Plan for reducing latency and improving perceived smoothness in the Nomicous
web app. Decisions from a grill-with-docs session (July 2026).

**Current stack:** Next.js App Router (the migration record is retained in
[`nextjs-migration.md`](nextjs-migration.md)).

Guidance maps to [Vercel React Best Practices](https://github.com/vercel-labs/agent-skills/tree/main/skills/react-best-practices)
rule prefixes (`async-`, `bundle-`, `server-`, `client-`, `rerender-`, etc.).

---

## Goals

- Eliminate repeated full-resolution image downloads when navigating between the
  document part list and the page editor.
- Reduce time-to-interactive on heavy routes (page editor, document detail).
- Make scrolling, zooming, and pairing feel responsive under load.
- Preserve authenticated access to protected media without sacrificing cacheability
  where possible.

---

## Current bottlenecks

### 1. Images are re-fetched on every mount

`AuthenticatedImage` (`nomicous/frontend/src/components/AuthenticatedImage.tsx`)
fetches protected media with a JWT `Authorization` header, converts the response to
a blob, and creates an object URL. On unmount it **revokes** that object URL.

Navigating document list → editor → back downloads the same page image again.
There is no cross-navigation or cross-component cache.

### 2. JSON deduplication does not cover images

`getCache.ts` (`dedupedGet`) deduplicates in-flight **JSON** GET requests only.
Binary image fetches bypass this layer entirely (`fetchBinaryApi` in `client.ts`).

### 3. N+1 image requests on document pages

`PartList` renders one `AuthenticatedImage` per document part (48×64px CSS thumb).
A document with many pages triggers many parallel **full-size** downloads.

### 4. Inference hooks re-fetch the same image

`usePairingState` and `useLayoutMutations` call `fetchBinaryApi(partImageUrl)` to
produce base64 for local inference, independent of whatever `AuthenticatedImage`
already loaded for display.

### 5. No HTTP caching on image responses

`part_image_response` in `nomicous/backend/document/api/media_responses.py`
returns raw bytes with no `Cache-Control` or `ETag`.

### 6. Route-level code splitting

**Next.js App Router** splits routes automatically. Keep editor-only dependencies
out of shared layouts so that route boundary remains effective.

### 7. List pages re-fetch on every navigation

`ProjectsPage`, `ProjectDashboardPage`, and `DocumentDetailPage` each call
`api.me()` and reload data in `useEffect` with no stale-while-revalidate cache.

### 8. WebP storage

Uploads are normalized to WebP in `encoding.py` (`encode_part_image`). Default is
**lossless** WebP (`MEDIA_WEBP_LOSSLESS=true`). Display thumbnails are the bigger
lever than re-encoding uploads.

---

## Locked decisions (grill session)

### Phase 0 — Current performance work

| # | Decision | Choice |
|---|----------|--------|
| P0-1 | Client image cache | **In-memory session `Map`** — no IndexedDB v1 |
| P0-2 | Cache key (now) | **Normalized path** (`/media/parts/{id}`); clear all on logout |
| P0-3 | Cache key (long-term) | Evolve to **`partId + variant + revision`** when thumbs + auth land |
| P0-4 | Backend headers | **Separate PR first** — `Cache-Control` + `ETag` on media routes |
| P0-5 | Frontend cache | **Separate PR** — `imageCache.ts` + wire AuthenticatedImage + inference |
| P0-6 | Thumbnails | **`GET /media/parts/{id}?w=200`** (and same on `/public/media/...`) |
| P0-7 | Thumb encoding | **Lossy WebP ~q85** on `?w=` path only; full route serves stored lossless bytes |
| P0-8 | Prefetch | **Hover + focus** on part row, **~100ms debounce** → prefetch full image into cache |
| P0-9 | Ship order | **Optimize the existing Next.js app first**, then layer follow-up refinements |

### Phase 3 — After Next migration

| # | Decision | Choice |
|---|----------|--------|
| P3-1 | List data fetching | **SWR** for `me`, projects, documents, document detail |
| P3-2 | Editor document reuse | **Shared SWR key** `['document', projectId, documentId]` on detail + editor pages |
| P3-3 | Rendering (Tier 4 subset) | **Memo `PartRow` + `AuthenticatedImage`**, **`useDeferredValue`** on canvas overlays during zoom/pan, **`content-visibility: auto`** on `.part-row` |
| P3-4 | Tier 4 remainder | **Defer** `startTransition` for jobs and passive scroll listeners unless profiling shows need |
| P3-5 | Auth + Tier 5 | **In-memory access token + HttpOnly refresh cookie + CSRF** — last in Phase 3 (see [Phase 3d](#phase-3d--auth--tier-5)) |
| P3-6 | E2E | **No Playwright** for now — Vitest + manual smoke |

---

## Implementation roadmap

### Phase 0 — Image performance (highest ROI)

```
PR 1  Backend: ETag + Cache-Control on GET /media/parts/{part_id}
      and GET /public/media/parts/{part_id}
      ETag from part.image_key (DocumentPart has no updated_at)
      Cache-Control: private, max-age=86400 (tune as needed)

PR 2  Frontend: imageCache.ts
      - ref-counted blob URLs
      - dedupedFetchBinary(url)
      - clearImageCache() on logout; invalidatePartImage(partId) on delete
      - Wire AuthenticatedImage, fetchBinaryApi, usePairingState, useLayoutMutations

PR 3  Backend: resize when ?w= present (PIL already in encoding.py)
      - lossy WebP q≈85 for w= query; full bytes when omitted
      Frontend: PartList src={url + '?w=200'}
      - prefetchPartImage on PartRow mouseEnter/focus (100ms debounce)
```

**Files:**

- `nomicous/frontend/src/api/imageCache.ts` (new)
- `nomicous/frontend/src/components/AuthenticatedImage.tsx`
- `nomicous/frontend/src/api/client.ts`
- `nomicous/frontend/src/components/page-editor/hooks/usePairingState.ts`
- `nomicous/frontend/src/components/page-editor/hooks/useLayoutMutations.ts`
- `nomicous/frontend/src/components/document/PartList.tsx`
- `nomicous/frontend/src/auth/storage.ts` (call clearImageCache on logout)
- `nomicous/backend/document/api/media_responses.py`
- `nomicous/backend/document/api/media.py`
- `nomicous/backend/document/api/public_media.py`

### Phase 1 — Next.js refinements

The Next.js migration is complete. Preserve the Phase 0 client-component boundaries
while applying these refinements.

### Phase 2 — Free wins from Next

- Route-level code splitting (automatic per `app/` segment)
- Optional: `loading.tsx` skeletons per route
- Bundle analyzer on `next build`

### Phase 3a — SWR

```ts
// Examples — implement as hooks in src/api/ or src/hooks/
useSWR('me', () => api.me())
useSWR('projects', () => api.listProjects())
useSWR(['project', projectId], () => api.getProject(projectId))
useSWR(['documents', projectId, includeArchived], ...)
useSWR(['document', projectId, documentId], () => api.getDocument(...))
```

- Editor `usePageEditorData`: read document from shared SWR key before fetching
- Keep `dedupedGet` for one-off JSON calls if needed
- `mutate()` after create/delete/upload/reorder

**Tier 3 #13 (parallel image in editor):** start `prefetchPartImage` as soon as
`part` is known in editor — do not wait for layout/lines JSON.

### Phase 3b — Rendering

| Item | Action |
|------|--------|
| Memo | `React.memo(PartRow)`, `React.memo(AuthenticatedImage)` with stable props |
| Deferred overlays | `useDeferredValue(lines)` (or layout) in `PageEditorCanvas` during zoom |
| Long lists | `.part-row { content-visibility: auto; contain-intrinsic-size: ... }` |

### Phase 3c — Bundle

- Import Ant Design from direct paths (`antd/es/spin`) where tree-shaking leaves fat
- `next.config` `experimental.optimizePackageImports: ['antd']` if available
- Run `@next/bundle-analyzer` once after migration

### Phase 3d — Auth + Tier 5

**Approach (locked):** in-memory access tokens + HttpOnly refresh/session cookie +
CSRF. ADR: [`docs/adr/0001-browser-auth-memory-cookie-csrf.md`](../adr/0001-browser-auth-memory-cookie-csrf.md).
Full flow: [`nextjs-migration.md` — Auth architecture](nextjs-migration.md#auth-architecture-phase-3d).

Ship after Phase 3a–c. PR sequence:

```
PR A  Backend: refresh cookie on login/register; POST /auth/refresh, /auth/logout;
      get_current_user accepts Bearer OR cookie; CSRF on mutations; CORS credentials

PR B  Frontend: AuthProvider (memory token, bootstrap refresh on load);
      remove localStorage; credentials: 'include'; CSRF header on mutations;
      clearImageCache() on logout

PR C  Media: get_current_user on GET /media/* via cookie;
      AuthenticatedImage → <img> or next/image; evolve cache keys to partId+variant+revision

PR D  Optional: Next middleware forwards cookie to /me; service worker on /media/*
```

| Item | Action |
|------|--------|
| Access token | Memory only — never `localStorage` |
| Session | HttpOnly cookie on `api.nomicous.com`, `Domain=.nomicous.com` |
| CSRF | Required on mutating API calls |
| Display images | `<img>` / `next/image` after cookie works on media routes |
| Inference | Keep in-memory blob cache for base64 dedup |

---

## Cache design notes

### In-memory cache (Phase 0)

```ts
type CacheEntry = {
  blob: Blob;
  objectUrl: string;
  refCount: number;
};

// Key now: normalizePath('/media/parts/{uuid}')
// Key later: `${partId}:full:${revision}` | `${partId}:w200:${revision}`
```

Part images are **immutable per `part_id`** today (no replace-upload endpoint).
`image_url` is always `/media/parts/{part_id}` with no version param.

### Thumbnail sizing

Part list thumb CSS: **48×64px** (`theme-shell.css`). `w=200` gives ~2× retina
headroom. Blur-up placeholder (Tier 4 #21) can use `?w=64` later.

---

## Verification

| Metric | How |
|--------|-----|
| Repeat navigation to same part | Network tab: image request count → 0 (cache hit) |
| Document with N parts | Thumbs use `?w=200`; count bytes per thumb vs full |
| Back from editor | No full image re-download |
| Local inference | One fetch; pairing/layout read from cache |
| List back-navigation | SWR shows stale data instantly, revalidates in background |
| Zoom/pan with many segments | Performance panel: fewer layout/paint during pan |

Manual smoke: same path as [`nextjs-migration.md`](nextjs-migration.md#manual-smoke-path-definition-of-done).

---

## Open decisions

### Auth implementation details (TBD at build time)

Approach is locked (memory + cookie + CSRF). Still to decide when implementing:

- Access vs refresh TTL values
- CSRF token shape (double-submit vs header token)
- Cookie names and rotation policy

Until Phase 3d ships: keep `localStorage` Bearer + `AuthenticatedImage` blob cache.

### IndexedDB (rejected for v1)

Revisit only if hard-refresh repeat visits become a measured problem.

### Playwright E2E (deferred)

Add when auth or image caching makes regressions harder to catch manually.

---

## Related docs

- [`nextjs-migration.md`](nextjs-migration.md) — framework migration spec
- [`nomicous/frontend/README.md`](../../nomicous/frontend/README.md) — frontend setup
- [`nomicous/CONTEXT.md`](../../nomicous/CONTEXT.md) — domain glossary (not implementation)
- [`deployment/production.md`](../deployment/production.md) — Vercel hosting
