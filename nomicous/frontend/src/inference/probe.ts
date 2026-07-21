import { HELPER_PROBE_TIMEOUT_MS } from "./constants";
import { fetchHelper } from "./helperClient";

export async function probeHelperHealth(): Promise<boolean> {
  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    HELPER_PROBE_TIMEOUT_MS,
  );
  try {
    const response = await fetchHelper("/health", {
      method: "GET",
      signal: controller.signal,
    });
    if (!response.ok) return false;
    const body = (await response.json()) as { status?: string };
    return body.status === "ok";
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeout);
  }
}
