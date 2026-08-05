import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  queryClient,
  taggedMeta,
  type ResourceKey,
  type ResourceTag,
} from "../api/queryClient";

/**
 * `E` is the shape a call site wants its failures in. Most want the banner
 * string they already had; the public document page distinguishes "not published
 * or does not exist" from "the read failed", so it uses a union instead.
 */
/** Cache slot for a query whose key is not known yet; never fetched. */
const DISABLED_KEY: ResourceKey = ["__disabled__"];

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
 * One read of server state.
 *
 * The lifecycle - in-flight sharing, a freshness window, cancellation on
 * unmount, and a refetch that cannot land out of order - is React Query's. What
 * stays here is the part that is this app's: failures are mapped to a call
 * site's own banner text by `onError`, `data` drops to null on failure rather
 * than showing the last good value beside an error, and tags travel in `meta` so
 * a write can invalidate a composite read without knowing its key.
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
  // Held in refs so a call site may pass inline closures without restarting the
  // read on every render. The key is what decides when a read is stale.
  const readRef = useRef(read);
  readRef.current = read;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  // Call sites build their key inline, so its identity changes every render.
  // Reading it from a ref keeps `refetch` and `patch` stable, as they were when
  // this hook owned its own cache.
  const keyRef = useRef(key);
  keyRef.current = key;

  const query = useQuery<T>(
    {
      // A disabled query never fetches, but it still occupies a cache entry, so
      // it gets its own key rather than sharing one empty key with every other.
      queryKey: key ?? DISABLED_KEY,
      queryFn: () => readRef.current(),
      meta: taggedMeta(tags),
      enabled: key !== null,
    },
    // The client is passed rather than taken from context, because the writes
    // that invalidate these reads are not components either - a session ends in
    // `storage.ts`, a mutation reports itself through `invalidateAfter` - and
    // they all have to reach the same cache.
    queryClient,
  );

  // `onError` both maps the failure and performs the call site's side effect (a
  // login redirect, a toast), so it runs once per failure in an effect rather
  // than during render. `errorUpdatedAt` changes on every failure, including a
  // repeat of an identical one.
  const [error, setError] = useState<E | null>(null);
  const { isError, errorUpdatedAt } = query;
  const failureRef = useRef<unknown>(null);
  failureRef.current = query.error;
  useEffect(() => {
    if (!isError) {
      setError(null);
      return;
    }
    setError(onErrorRef.current(failureRef.current));
    // `errorUpdatedAt` advances on every failure, including a repeat of an
    // identical one and the first failure under a new key.
  }, [isError, errorUpdatedAt]);

  const refetch = useCallback(async () => {
    const currentKey = keyRef.current;
    if (currentKey === null) return;
    await queryClient.refetchQueries({ queryKey: currentKey, exact: true });
  }, []);

  const patch = useCallback((update: (current: T) => T) => {
    const currentKey = keyRef.current;
    if (currentKey === null) return;
    queryClient.setQueryData<T>(currentKey, (current) =>
      current === undefined ? current : update(current),
    );
  }, []);

  return {
    // A failed read shows its banner alone; the last good value is not left on
    // screen beside it.
    data: isError ? null : (query.data ?? null),
    // First load only. A background refetch on window focus must not put the
    // public document page back to its skeleton, and callers that want a busy
    // flag across an explicit `refetch` already keep their own.
    loading: key === null || query.isPending,
    error,
    refetch,
    patch,
  };
}
