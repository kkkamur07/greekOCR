import { HELPER_BASE_URLS } from "./constants";

export async function fetchHelper(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  let lastError: unknown;

  for (const baseUrl of HELPER_BASE_URLS) {
    try {
      return await fetch(`${baseUrl}${path}`, init);
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
