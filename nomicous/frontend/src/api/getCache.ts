const inFlight = new Map<string, Promise<unknown>>();

/** Share one in-flight GET per key so Strict Mode / concurrent mounts do not duplicate requests. */
export function dedupedGet<T>(key: string, fetcher: () => Promise<T>): Promise<T> {
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
