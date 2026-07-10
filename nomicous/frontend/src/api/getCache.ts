const inFlight = new Map<string, Promise<unknown>>();
let authVersion = 0;

/** Share one in-flight GET per key so Strict Mode / concurrent mounts do not duplicate requests. */
export function dedupedGet<T>(
  key: string,
  fetcher: () => Promise<T>,
): Promise<T> {
  const existing = inFlight.get(key);
  if (existing) {
    return existing as Promise<T>;
  }

  const promise = fetcher().finally(() => {
    inFlight.delete(key);
  });
  inFlight.set(key, promise);
  return promise;
}

/** Invalidate all authenticated GET work after login, refresh, or logout. */
export function invalidateAuthGetCache(): number {
  authVersion += 1;
  inFlight.clear();
  return authVersion;
}

export function getAuthRequestVersion(): number {
  return authVersion;
}
