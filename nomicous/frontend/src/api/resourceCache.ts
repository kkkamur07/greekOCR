/**
 * Retained server-state cache.
 *
 * `dedupedGet` in the HTTP layer already shares an *in-flight* GET, which is
 * what stops a Strict Mode double mount from issuing the same request twice. It
 * deliberately forgets everything the moment that request settles, so
 * navigating away and back refetches a page that has not changed. This module
 * is the retained half: a settled read stays readable until it ages out or a
 * write declares it stale.
 *
 * Invalidation is by tag rather than by key because most reads are composites -
 * a dashboard loads its project and that project's documents in one pass - and a
 * write to either part has to make the composite stale. A write names the tag it
 * dirtied; it never enumerates the reads that happen to depend on it. That is
 * what keeps invalidation declared once per resource instead of remembered at
 * every call site. The vocabulary itself lives in `resources.ts`.
 */

/** Identity of one cached read. */
export type ResourceKey = readonly (string | number | boolean)[];

/** What a read depends on, and therefore what a write can make stale. */
export type ResourceTag = string;

/**
 * How long a settled read stays servable without going back to the server.
 *
 * The app invalidates its own writes explicitly, so this window only bounds how
 * stale another client's write can leave the page - long enough that paging back
 * and forth costs nothing, short enough that a collaborator's change shows up
 * without a reload.
 */
export const RESOURCE_FRESH_MS = 30_000;

type CacheEntry = {
  tags: readonly ResourceTag[];
  request: Promise<unknown>;
  /** Set once `request` fulfils, so a remount can render without a loading pass. */
  settled: { data: unknown; at: number } | null;
};

const entries = new Map<string, CacheEntry>();
const listeners = new Map<string, Set<() => void>>();

/**
 * JSON rather than a joined string, so that no separator has to be assumed
 * absent from ids: `["a", "b"]` and `["a b"]` are different keys either way.
 */
export function serializeResourceKey(key: ResourceKey): string {
  return JSON.stringify(key);
}

function isFresh(settled: CacheEntry["settled"], now: number): boolean {
  return settled !== null && now - settled.at < RESOURCE_FRESH_MS;
}

function startRead<T>(
  serializedKey: string,
  tags: readonly ResourceTag[],
  read: () => Promise<T>,
): Promise<T> {
  const pending = read();
  const entry: CacheEntry = { tags, request: pending, settled: null };
  entries.set(serializedKey, entry);

  const tracked = pending.then(
    (data) => {
      // An invalidation or a newer read may have replaced this entry while the
      // request was in flight; only the entry that still owns it may record it.
      if (entries.get(serializedKey) === entry) {
        entry.settled = { data, at: Date.now() };
      }
      return data;
    },
    (error: unknown) => {
      // Failures are never retained: the next mount has to be free to retry.
      if (entries.get(serializedKey) === entry) {
        entries.delete(serializedKey);
      }
      throw error;
    },
  );
  entry.request = tracked;
  return tracked;
}

/**
 * Read a resource through the cache.
 *
 * An in-flight request is always shared, whatever `force` says - two components
 * asking for the same thing at the same time is exactly the duplicate this layer
 * exists to remove. `force` only overrides the freshness window, which is what an
 * explicit reload after a mutation needs.
 */
export function readResource<T>(
  key: ResourceKey,
  tags: readonly ResourceTag[],
  read: () => Promise<T>,
  options?: { force?: boolean },
): Promise<T> {
  const serializedKey = serializeResourceKey(key);
  const existing = entries.get(serializedKey);
  if (existing) {
    if (existing.settled === null) {
      return existing.request as Promise<T>;
    }
    if (!options?.force && isFresh(existing.settled, Date.now())) {
      return existing.request as Promise<T>;
    }
  }
  return startRead(serializedKey, tags, read);
}

/** Synchronously readable cached value, or null when a read is required. */
export function peekResource<T>(key: ResourceKey): { data: T } | null {
  const settled = entries.get(serializeResourceKey(key))?.settled ?? null;
  if (settled === null || !isFresh(settled, Date.now())) return null;
  return { data: settled.data as T };
}

/**
 * Fold a value the app already holds into an existing cached read.
 *
 * This is the optimistic path: a mutation that returns the updated object knows
 * more than the cache does, and re-reading it would be a round trip for data
 * already in hand. It deliberately does not create an entry - if the read has
 * been invalidated, the next reader should go to the server rather than adopt a
 * fragment - and it deliberately does not notify, because the caller is already
 * rendering the value it just wrote.
 */
export function patchResource<T>(key: ResourceKey, data: T): void {
  const entry = entries.get(serializeResourceKey(key));
  if (!entry || entry.settled === null) return;
  entry.settled = { data, at: entry.settled.at };
  entry.request = Promise.resolve(data);
}

/**
 * Drop every cached read carrying one of `tags` and tell whoever is showing it.
 *
 * Callers should not reach for this directly - `invalidateAfter` in
 * `resources.ts` names the writes, so the mapping from a write to the reads it
 * dirties is written down once.
 */
export function invalidateTags(tags: readonly ResourceTag[]): void {
  if (tags.length === 0) return;
  const dirty = new Set(tags);
  const invalidated: string[] = [];
  for (const [serializedKey, entry] of entries) {
    if (entry.tags.some((tag) => dirty.has(tag))) {
      entries.delete(serializedKey);
      invalidated.push(serializedKey);
    }
  }
  for (const serializedKey of invalidated) {
    for (const listener of listeners.get(serializedKey) ?? []) {
      listener();
    }
  }
}

/** Notifies a mounted reader that its entry went stale under it. */
export function subscribeToResource(
  key: ResourceKey,
  onInvalidated: () => void,
): () => void {
  const serializedKey = serializeResourceKey(key);
  const registered = listeners.get(serializedKey) ?? new Set<() => void>();
  registered.add(onInvalidated);
  listeners.set(serializedKey, registered);
  return () => {
    registered.delete(onInvalidated);
    if (registered.size === 0) {
      listeners.delete(serializedKey);
    }
  };
}

/**
 * Forget everything. Belongs at a session boundary - a login or a logout makes
 * every cached read someone else's data - and between tests.
 */
export function clearResourceCache(): void {
  entries.clear();
}
