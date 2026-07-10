import { HELPER_BASE_URL } from "./constants";

type CacheStatusResponse = {
  registry_model_id: string;
  registry_tag: string;
  cached: boolean;
};

/**
 * Ask the local helper whether a model's weights are already on disk.
 * Used to decide whether the first-time "Downloading…" banner should appear.
 * Returns `null` when the helper is unreachable so we do not flash a false
 * download banner for bundled models (e.g. Kraken segment).
 */
export async function fetchLocalCacheStatus(
  registryModelId: string,
  registryTag = "stable",
): Promise<boolean | null> {
  try {
    const params = new URLSearchParams({
      registry_model_id: registryModelId,
      registry_tag: registryTag,
    });
    const response = await fetch(
      `${HELPER_BASE_URL}/inference/v1/cache-status?${params.toString()}`,
      {
        method: "GET",
      },
    );
    if (!response.ok) return null;
    const body = (await response.json()) as CacheStatusResponse;
    return body.cached === true;
  } catch {
    return null;
  }
}
