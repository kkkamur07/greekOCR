type CacheEntry = {
  blob: Blob;
  objectUrl: string;
  references: number;
};

/** Insertion order doubles as LRU order - see `touch`. */
const entries = new Map<string, CacheEntry>();
const pending = new Map<string, Promise<CacheEntry>>();
/**
 * Paths an `acquirePartImage` call is currently waiting on.
 *
 * A caller cannot take a reference until its fetch resolves, so between an
 * entry being inserted and its own caller running `references += 1` there is a
 * microtask window in which the entry looks unreferenced. A sibling request
 * resolving inside that window would evict it and revoke an object URL that is
 * about to become the `src` of a live img, which is why a document with more
 * pages than the cache bound would show broken images for an arbitrary subset
 * of its thumbnails after a reload. A claim is registered synchronously, before
 * any await, so the window no longer exists.
 */
const claims = new Map<string, number>();
let cacheGeneration = 0;

/**
 * A manuscript page is megabytes of decoded image, so the cache is bounded:
 * paging through a long document must not grow the tab's memory without limit.
 * The bound covers a page plus its neighbours in both directions, at full size
 * and as thumbnails, so ordinary back-and-forth reading still hits the cache.
 */
export const MAX_CACHED_PART_IMAGES = 12;

const PART_IMAGE_PATH = /^\/(?:public\/)?media\/parts\/[^/]+$/;
const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";
const apiOrigin = new URL(apiBaseUrl).origin;

function isLocalHost(hostname: string): boolean {
  return (
    hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1"
  );
}

function isPermittedOrigin(url: URL): boolean {
  const api = new URL(apiBaseUrl);
  if (url.origin !== apiOrigin) return false;
  return (
    api.protocol === "https:" ||
    (api.protocol === "http:" && isLocalHost(api.hostname))
  );
}

/**
 * Normalize a document-part media URL to the API-relative form expected by
 * fetchBinaryApi. Query parameters are part of the representation cache key.
 */
export function normalizePartImagePath(src: string): string | null {
  try {
    const url = new URL(src, `${apiBaseUrl}/`);
    if (!isPermittedOrigin(url) || !PART_IMAGE_PATH.test(url.pathname))
      return null;

    const width = url.searchParams.get("w");
    if (
      width !== null &&
      (!/^[1-9]\d*$/.test(width) || url.searchParams.size !== 1)
    ) {
      return null;
    }
    if (width === null && url.searchParams.size !== 0) return null;

    return `${url.pathname}${width === null ? "" : `?w=${width}`}`;
  } catch {
    return null;
  }
}

/**
 * Mark `path` as most recently used. `Map` iterates in insertion order, so
 * re-inserting an entry at the end makes the eviction sweep read LRU order
 * straight off the map instead of tracking timestamps alongside it.
 */
function touch(path: string, entry: CacheEntry): CacheEntry {
  entries.delete(path);
  entries.set(path, entry);
  return entry;
}

/**
 * Drop least-recently-used entries until the cache is back inside its bound.
 *
 * An entry the UI is still showing, or is about to show, is never dropped: its
 * object URL is the `src` of a live <img>, and revoking it would blank the
 * image. That covers entries with live references and entries a caller has
 * claimed but not yet referenced. The cache can
 * therefore sit above its bound while many images are on screen at once, and
 * shrinks again as they are released.
 */
function evictLeastRecentlyUsed(): void {
  if (entries.size <= MAX_CACHED_PART_IMAGES) return;
  for (const [path, entry] of entries) {
    if (entries.size <= MAX_CACHED_PART_IMAGES) return;
    if (entry.references > 0 || claims.has(path)) continue;
    URL.revokeObjectURL(entry.objectUrl);
    entries.delete(path);
  }
}

function claim(path: string): void {
  claims.set(path, (claims.get(path) ?? 0) + 1);
}

function releaseClaim(path: string): void {
  const remaining = (claims.get(path) ?? 0) - 1;
  if (remaining > 0) claims.set(path, remaining);
  else claims.delete(path);
}

/**
 * The API client is imported lazily to keep it out of the cache module's own
 * import graph, but a page list mounting eighteen thumbnails at once would
 * otherwise start eighteen dynamic imports of it. One memoized promise is
 * enough, and it means the first thumbnail's import is the only one anything
 * waits on.
 */
let clientModule: Promise<typeof import("./client")> | null = null;

function loadClient(): Promise<typeof import("./client")> {
  // A rejected promise is neither null nor undefined, so memoizing one would
  // pin every later call to the same failure and break image loading for the
  // life of the page. Dynamic imports do fail in production: a deploy can
  // invalidate the chunk hash an open tab still points at. Forget the failure
  // so the next caller gets a fresh attempt.
  clientModule ??= import("./client").catch((error: unknown) => {
    clientModule = null;
    throw error;
  });
  return clientModule;
}

async function getEntry(path: string): Promise<CacheEntry> {
  const cached = entries.get(path);
  if (cached) return touch(path, cached);

  let request = pending.get(path);
  if (!request) {
    const generation = cacheGeneration;
    request = loadClient()
      .then(({ fetchBinaryApi }) => fetchBinaryApi(path))
      .then((blob) => {
        if (generation !== cacheGeneration) {
          throw new Error(
            "Image cache was cleared while the request was in flight.",
          );
        }
        const entry = {
          blob,
          objectUrl: URL.createObjectURL(blob),
          references: 0,
        };
        entries.set(path, entry);
        evictLeastRecentlyUsed();
        return entry;
      });
    pending.set(path, request);
    void request.finally(() => pending.delete(path)).catch(() => undefined);
  }
  return request;
}

export async function fetchPartImage(pathOrUrl: string): Promise<Blob> {
  const path = normalizePartImagePath(pathOrUrl);
  if (!path) throw new Error("Invalid protected part-image URL.");
  return (await getEntry(path)).blob;
}

export async function acquirePartImage(
  pathOrUrl: string,
): Promise<{ objectUrl: string; release: () => void }> {
  const path = normalizePartImagePath(pathOrUrl);
  if (!path) throw new Error("Invalid protected part-image URL.");

  // Claimed before the await, so no sweep can evict this path while the fetch
  // is in flight or while the resolution is still queued behind other work.
  claim(path);
  let entry: CacheEntry;
  try {
    entry = await getEntry(path);
    entry.references += 1;
  } finally {
    releaseClaim(path);
  }

  let released = false;
  return {
    objectUrl: entry.objectUrl,
    release: () => {
      if (released) return;
      released = true;
      entry.references = Math.max(0, entry.references - 1);
      // Releasing the last reference may be what lets the cache shrink back
      // inside its bound.
      evictLeastRecentlyUsed();
    },
  };
}

export function prefetchPartImage(pathOrUrl: string): void {
  const path = normalizePartImagePath(pathOrUrl);
  if (path) void getEntry(path).catch(() => undefined);
}

export function invalidatePartImage(partId: string): void {
  for (const [path, entry] of entries) {
    const pathPartId = path.split("?")[0].split("/").at(-1);
    if (pathPartId === partId) {
      URL.revokeObjectURL(entry.objectUrl);
      entries.delete(path);
    }
  }
}

export function clearImageCache(): void {
  cacheGeneration += 1;
  for (const entry of entries.values()) {
    URL.revokeObjectURL(entry.objectUrl);
  }
  entries.clear();
  pending.clear();
}
