import { HELPER_BASE_URLS } from "./constants";

/** Chromium Local Network Access: mark loopback so HTTPS pages can talk to the helper. */
type LoopbackRequestInit = RequestInit & {
  targetAddressSpace?: "loopback" | "local" | "public";
};

export async function fetchHelper(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  let lastError: unknown;
  const loopbackInit: LoopbackRequestInit = {
    ...init,
    targetAddressSpace: "loopback",
  };

  for (const baseUrl of HELPER_BASE_URLS) {
    try {
      return await fetch(`${baseUrl}${path}`, loopbackInit);
    } catch (error) {
      if (init?.signal?.aborted) {
        throw error;
      }
      lastError = error;
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Inference helper is unreachable.");
}
