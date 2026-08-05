import { useCallback, useEffect, useRef, useState } from "react";
import {
  patchResource,
  peekResource,
  readResource,
  serializeResourceKey,
  subscribeToResource,
  type ResourceKey,
  type ResourceTag,
} from "../api/resourceCache";

/**
 * `E` is the shape a call site wants its failures in. Most want the banner
 * string they already had; the public document page distinguishes "not published
 * or does not exist" from "the read failed", so it uses a union instead.
 */
export type ServerQuery<T, E = string> = {
  data: T | null;
  loading: boolean;
  error: E | null;
  /** Re-reads past the freshness window. Awaitable, so a caller can keep a busy flag up across the reload. */
  refetch: () => Promise<void>;
  /**
   * Folds a mutation's own response into the view without a round trip, writing
   * through to the cache so a remount does not show the pre-mutation value.
   */
  patch: (update: (current: T) => T) => void;
};

export type ServerQueryOptions<T, E = string> = {
  /**
   * Identity of this read, and the cache entry it fills.
   *
   * `null` disables the query: nothing is fetched and no state transition
   * happens at all. That is what the page loaders' early `return`s did before -
   * a missing route param or an absent access token leaves `loading` true while
   * a redirect is under way, rather than flashing an empty page.
   */
  key: ResourceKey | null;
  /** What this read depends on, so a write can declare it stale. */
  tags: readonly ResourceTag[];
  read: () => Promise<T>;
  /**
   * Turns a failed read into this call site's banner text, and performs whatever
   * else that call site does on failure - a login redirect, a toast.
   *
   * The rules genuinely differ between pages: some branch on `ApiError` so a
   * network `TypeError` yields a generic sentence, others branch on `Error` and
   * surface `err.message`, and only some rewrite 403/404 into "not available to
   * your account". Flattening them here would silently change what several pages
   * say, so the mapping stays at the call site. Returning `null` means the
   * failure was handled without a banner.
   */
  onError: (error: unknown) => E | null;
};

/**
 * One read of server state, with the lifecycle that used to be hand-rolled at
 * every call site: a loading flag, an error string, cancellation on unmount, and
 * a refetch that cannot land out of order.
 *
 * This hook only ever *reads*. It has no mutation entry point and no way to be
 * handed one, which is what stops a failing refresh from re-entering a write
 * path - see `runLocalFirstWrite` for the other half of that separation.
 */
export function useServerQuery<T, E = string>({
  key,
  tags,
  read,
  onError,
}: ServerQueryOptions<T, E>): ServerQuery<T, E> {
  const serializedKey = key === null ? null : serializeResourceKey(key);

  // A warm cache entry is applied before the first paint, so returning to a page
  // does not flash its skeleton at data that is already known to be current.
  const [data, setData] = useState<T | null>(
    () => (key === null ? null : peekResource<T>(key)?.data) ?? null,
  );
  const [loading, setLoading] = useState(
    () => key === null || peekResource(key) === null,
  );
  const [error, setError] = useState<E | null>(null);

  // Held in refs so a call site may pass inline closures without restarting the
  // read on every render. The key is what decides when a read is stale.
  const keyRef = useRef(key);
  keyRef.current = key;
  const tagsRef = useRef(tags);
  tagsRef.current = tags;
  const readRef = useRef(read);
  readRef.current = read;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const mountedRef = useRef(true);
  // Only the newest attempt may write state: a slow response for a key the
  // component has already navigated away from must not overwrite the new one.
  const attemptRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const runRead = useCallback(async (options?: { force?: boolean }) => {
    const currentKey = keyRef.current;
    if (currentKey === null) return;

    attemptRef.current += 1;
    const attempt = attemptRef.current;
    const isCurrent = () => mountedRef.current && attemptRef.current === attempt;

    setLoading(true);
    setError(null);
    try {
      const value = await readResource<T>(
        currentKey,
        tagsRef.current,
        () => readRef.current(),
        options,
      );
      if (!isCurrent()) return;
      setData(value);
    } catch (failure) {
      if (!isCurrent()) return;
      setData(null);
      setError(onErrorRef.current(failure));
    } finally {
      if (isCurrent()) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (serializedKey === null) return;
    const currentKey = keyRef.current;
    if (currentKey === null) return;

    const cached = peekResource<T>(currentKey);
    if (cached) {
      // Already current: adopt it without a loading pass or a request.
      attemptRef.current += 1;
      setData(cached.data);
      setError(null);
      setLoading(false);
    } else {
      void runRead();
    }

    // A write elsewhere in the app can drop this entry; re-read so two views of
    // the same resource cannot disagree.
    return subscribeToResource(currentKey, () => {
      void runRead();
    });
  }, [serializedKey, runRead]);

  const refetch = useCallback(() => runRead({ force: true }), [runRead]);

  const dataRef = useRef(data);
  dataRef.current = data;
  const patch = useCallback((update: (current: T) => T) => {
    const currentKey = keyRef.current;
    if (dataRef.current === null || currentKey === null) return;
    const next = update(dataRef.current);
    dataRef.current = next;
    setData(next);
    patchResource(currentKey, next);
  }, []);

  return { data, loading, error, refetch, patch };
}
