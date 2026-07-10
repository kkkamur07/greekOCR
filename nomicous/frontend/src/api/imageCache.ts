type CacheEntry = {
  blob: Blob;
  objectUrl: string;
  references: number;
};

const entries = new Map<string, CacheEntry>();
const pending = new Map<string, Promise<CacheEntry>>();
let cacheGeneration = 0;

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

async function getEntry(path: string): Promise<CacheEntry> {
  const cached = entries.get(path);
  if (cached) return cached;

  let request = pending.get(path);
  if (!request) {
    const generation = cacheGeneration;
    request = import("./client")
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
  const entry = await getEntry(path);
  entry.references += 1;

  let released = false;
  return {
    objectUrl: entry.objectUrl,
    release: () => {
      if (released) return;
      released = true;
      entry.references = Math.max(0, entry.references - 1);
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
